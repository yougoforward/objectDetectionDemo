#!/usr/bin/python2
# -*- coding: utf-8 -*-

# Created by: VPALab Hailong Zhu and Dayu Jia  with PyQt5 UI code generator 5.5.1
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import threading
import time
import Queue
import argparse
import math
import types
import copy
import logging
from PIL import Image, ImageDraw, ImageFont
sys.path.append('/home/long/github/caffe2/build')
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import utils.c2 as c2_utils
from collections import defaultdict
from utils.timer import Timer
import utils.logging
running = False
capture_thread = None
q = Queue.Queue(maxsize=3)
reload(sys)
sys.setdefaultencoding('utf-8')
c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--camera',
        dest='camera',
        help='input option(/path/to/video or camera index )',
        default='0',
        type=str
    )

    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='e2e_faster_rcnn_R-50-FPN_2x.yaml',
	    # default='/home/long/github/detectron/configs/12_2017_baselines/retinanet_R-50-FPN_2x.yaml',
        #default = '/home/long/github/detectron/configs/getting_started/e2e_lighthead_rcnn_R-50-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='e2e_faster_rcnn_R-50-FPN_2x.pkl',
        # default = '/media/E/models/detectron/retinanet_R-50-FPN_2x.pkl',
        #default='/media/E/models/detectron/compactfasterfpn50.pkl',
        type=str
    )

    return parser.parse_args()


def get_class_string(class_index, class_names):
    class_text = class_names[class_index] if class_names is not None else \
        'id{:d}'.format(class_index)
    return class_text

def vis_class(img, pos, class_str):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    theta, radius = computeaspect(pos)
    thetaText = u'%d度'%(90-theta)
    distText=u'%.2fm'%radius
    txt = class_str#+thetaText+distText
    #cv2 to pil
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    # draw pil
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    font = ImageFont.truetype("simsun.ttc", 12, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
    draw.text((x0, y0-15), txt, (0, 255, 0), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    # if radius*math.sin(theta)<6.5:
    #     draw.text(((x0+int(pos[2]))/2-20, y0 + 30), u'危险！', (255, 0, 0),
    #               font=ImageFont.truetype("../tools/simsun.ttc", 30, encoding="utf-8"))  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
    # pil to cv2
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return img


def vis_bbox(img, bbox, thick=1, color=_GREEN):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def computeaspect(bbox):
    """compute distance and aspect of the object ."""
    u, v = (bbox[0] + bbox[2]) / 2.0, bbox[3] # the cenetr location of the lower box boundary

    x = 0.0230 * u - ((0.9996 * u - 550.3179) * (37.6942 * v - 2.2244e+06)) / (
            1.6394e+03 * v - 4.1343e+05) - 12.9168
    y = ((0.0070 * u - 1.6439e+03) * (37.6942 * v - 2.2244e+06)) / (
            1.6394e+03 * v - 4.1343e+05) - 1.6046e-04 * u + 0.0902

    if y < 6500: # # dist is too little to be precise, recompute
        v = (bbox[2]-bbox[0])*1.5+bbox[1]
        x = 0.0230 * u - ((0.9996 * u - 550.3179) * (37.6942 * v - 2.2244e+06)) / (
                1.6394e+03 * v - 4.1343e+05) - 12.9168
        y = ((0.0070 * u - 1.6439e+03) * (37.6942 * v - 2.2244e+06)) / (
                1.6394e+03 * v - 4.1343e+05) - 1.6046e-04 * u + 0.0902

    theta = math.degrees(math.atan2(y, x))
    radius = math.sqrt(x ** 2 + y ** 2)/1000
    return theta, radius

def demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=[], show_box=False, show_class=False,
                                   class_names=[], color_list=[], cls_sel=[], frame=[]):
    """Constructs a numpy array with the detections visualized."""
    box_list = [b for b in [cls_boxes[i] for i in cls_sel] if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    classes = []

    for j in cls_sel:
        classes += [j] * len(cls_boxes[j])

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < min(thresh):
        return im
    # for each detected box
    for i, cls_id in enumerate(classes[0:]):
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh[cls_id]:
            continue

        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), color=color_list[cls_id])

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], class_names)
            im = vis_class(im, bbox, class_str)
        frame["img"] = im
    return frame

def camera(queue, width, height, fps, args):
    global running
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    cam = args.camera
    if '.' not in cam:
        cam = int(cam)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg()
    # dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    start_time = 0
    count = 0
    #class_names =[
    #    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #    'bus', 'train', 'truck']

    class_names = [
         '__background__', u'人', u'自行车', u'小汽车', u'摩托车', 'airplane',
         u'公共汽车', 'train', u'卡车']
    # color_list=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,255,0],[255,0,255],[255,255,255]]

    # class_names = [
    #     '__background__', u'人', u'自行车', u'车', u'摩托车', 'airplane',
    #     u'车', 'train', u'车']
    color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 255], [0, 0, 255],
                  [255, 0, 255], [0, 0, 255]]

    cls_sel = [1, 2, 3, 4, 6, 8]
    cls_thresh = [1, 0.5, 0.6, 0.8, 0.6, 0.9, 0.5, 0.9, 0.5]
    if count == 0:
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )
    capture = cv2.VideoCapture(cam)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    size = (float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(size)
    fps=capture.get(cv2.CAP_PROP_FPS)
    print("%f fps"%fps)
    print(type(cam))
    #record
    fourcc = cv2.VideoWriter_fourcc(b'X', b'V', b'I', b'D')
    videoWriter = cv2.VideoWriter('ObjectDetection.avi', fourcc, 10, (int(size[0]),int(size[1])))
    t1=0
    dett=0
    vt=0
    while (1):
        frame = {}
        readtime =time.time()
        ret, im = capture.read()
        print("read frame time %f.2 ms"%((time.time()-readtime)*1000))
        # if type(cam)==str:
        #    im=cv2.resize(im, None, None, fx= width/size[0], fy= height/size[1], interpolation=cv2.INTER_LINEAR)
        # ori_im = copy.deepcopy(im)
        frame["img"] = im
        # # detect one image
        if running == False:
            # if type(cam)==str:
            #     time.sleep(1/fps)
            count = 0
        else:
            if count == 2:
                start_time = time.time()
            count = count + 1
            print('remain time %f.2 ms'%((time.time()-t1-dett-vt)*1000))
            t1 = time.time()
            timers = defaultdict(Timer)
            with c2_utils.NamedCudaScope(0):
                cls_boxes, _, _ = infer_engine.im_detect_all(
                    model, im, None, timers=timers)
                dett = time.time()-t1
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
            demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                                            class_names=class_names, color_list=color_list, cls_sel=cls_sel, frame=frame)
            vt = time.time()-t1-dett
            print("det time %f.2ms, vis time %f.2ms"%(dett*1000,vt*1000))
            if count>=2:
                avg_fps = (count-1) / (time.time() - start_time)
                cv2.putText(frame["img"], '{:s} {:.2f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255),
                        lineType=cv2.LINE_AA)
            videoWriter.write(frame["img"])  # write frame to video
            # new_im = frame["img"]
            # combine=cv2.hconcat([ori_im, new_im])
            # frame["img"] = combine
        queue.put(frame)
    videoWriter.write(frame["img"])  # write frame to video



class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        window_width = self.frameSize().width()
        window_height = self.frameSize().height()
        parent_w = self.parentWidget().frameSize().width()
        parent_h = self.parentWidget().frameSize().height()
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        self.move((parent_w-window_width)/2, (parent_h-window_height)/2)
        qp.end()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        #screen resolution
        desktop = QtWidgets.QApplication.desktop()
        x = desktop.width()
        y = desktop.height()
        MainWindow.resize(x, y)

        x2=x/2560.0
        y2=y/1440.0
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(int(180*x2), int(160*y2), int(2000*x2), int(1200*y2)))

        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizentalLayout = QtWidgets.QHBoxLayout(self.verticalLayoutWidget)
        self.horizentalLayout.setObjectName("horizentalLayout")

        self.startButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.stopButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.exitButton = QtWidgets.QPushButton(self.verticalLayoutWidget)


        font = QtGui.QFont()
        font.setPointSize(15)
        #start button
        self.startButton.setFont(font)
        self.startButton.setObjectName("startButton")
        self.horizentalLayout.addWidget(self.startButton)
        self.horizentalLayout.addSpacing(50)
        #stop button
        self.stopButton.setFont(font)
        self.stopButton.setObjectName("stopButton")
        self.horizentalLayout.addWidget(self.stopButton)
        self.horizentalLayout.addSpacing(50)
        # exit button
        self.exitButton.setFont(font)
        self.exitButton.setObjectName("exitButton")
        self.horizentalLayout.addWidget(self.exitButton)

        self.verticalLayout.addLayout(self.horizentalLayout, stretch=0)
        # self.verticalLayout.addSpacing(5)

        # self.label = QtWidgets.QLabel(self.centralwidget)
        # # self.label.setGeometry(QtCore.QRect(0, 0, int(180*x2), int(180*y2)))
        # self.label.setObjectName("logo")
        # jpg = QtGui.QPixmap('logo3.jpg')
        # self.label.setPixmap(jpg)
        # MainWindow.setCentralWidget(self.centralwidget)

        self.groupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(0, int(25*y2), int(2000*x2), int(1200*y2)))
        self.widget.setObjectName("widget")
        self.verticalLayout.addWidget(self.groupBox)
        MainWindow.setCentralWidget(self.centralwidget)
        # self.menubar = QtWidgets.QMenuBar(MainWindow)
        # self.menubar.setGeometry(QtCore.QRect(0, 0, int(789*x2), int(25*y2)))
        # self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(int(600*x2), int(60*y2), int(1200*x2), int(91*y2)))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.label.setFont(font)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.startButton.clicked.connect(self.start_clicked)
        self.stopButton.clicked.connect(self.stop_clicked)
        self.exitButton.clicked.connect(self.exit_clicked)

        self.window_width = self.widget.frameSize().width()
        self.window_height = self.widget.frameSize().height()

        self.ImgWidget = OwnImageWidget(self.widget)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测系统"))
        self.startButton.setText(_translate("MainWindow", "开始检测"))
        self.stopButton.setText(_translate("MainWindow", "结束检测"))
        self.exitButton.setText(_translate("MainWindow", "退出"))
        self.groupBox.setTitle(_translate("MainWindow", ""))
        self.label.setText(_translate("MainWindow", "天津大学视觉模式分析实验室目标检测系统"))


    def start_clicked(self):
        global running
        running = True
        # capture_thread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('正在检测')

    def stop_clicked(self):
        global running
        running = False
        self.stopButton.setEnabled(False)
        self.stopButton.setText('正在结束')
        self.startButton.setEnabled(False)
        self.startButton.setText('正在结束检测')
        time.sleep(1)
        self.stopButton.setEnabled(True)
        self.stopButton.setText('结束检测')

        self.startButton.setEnabled(True)
        self.startButton.setText('开始检测')

    def exit_clicked(self):
        capture_thread.exit()



    def update_frame(self):
        if not q.empty():
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            self.ImgWidget.setImage(image)
            q.task_done()
    def closeEvent(self, event):
        global running
        running = False


if __name__ == "__main__":
    import sys
    capture_thread = threading.Thread(target=camera, args=(q, 800, 600, 30, parse_args()))
    capture_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

