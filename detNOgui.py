#!/usr/bin/python2
# -*- coding: utf-8 -*-

# Created by: VPALab Hailong Zhu and Dayu Jia  with PyQt5 UI code generator 5.5.1

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import threading
import time
import Queue
import argparse
import logging
import math
import types
import copy
from PIL import Image, ImageDraw, ImageFont
sys.path.append('/home/long/github/caffe2/build')
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import utils.c2 as c2_utils

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
# fisherROI = [[193, 173], [342, 110], [538, 97], [747, 148], [797, 173], [813, 290], [796, 388], [670, 425], [482, 439], [270, 413], [177,
#              360]]
fisherROI = [np.array([386, 346]), np.array([684, 220]), np.array([1076,  194]), np.array([1494,  296]), np.array([1594,  346]),
             np.array([1626,  580]), np.array([1592,  776]), np.array([1340,  850]), np.array([964, 878]), np.array([540, 826]), np.array([354, 720])]
fisherROI = np.array(fisherROI)
# fisherROI = [np.array(i)*2 for i in fisherROI]


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--camera',
        dest='camera',
        help='input option(/path/to/video or camera index )',
        default='/home/long/objectdetection/01010002855000000.mp4',
        type=str
    )

    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='fisher221_1gpu_e2e_faster_rcnn_R-50-FPN.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='model300_final.pkl',
        type=str
    )

    return parser.parse_args()


def get_class_string(class_index, class_names):
    class_text = class_names[class_index] if class_names is not None else \
        'id{:d}'.format(class_index)
    return class_text

def vis_class(img, pos, class_str, score):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    theta, radius = computeaspect(pos)
    thetaText = u'%d度'%(90-theta)
    distText=u'%.2fm'%radius
    txt = class_str+thetaText+distText
    txt = class_str+'  '+'%.2f'%score
    txt = '%.2f'%score
    # cv2 to pil
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pil_im = Image.fromarray(cv2_im)
    # draw pil
    draw = ImageDraw.Draw(pil_im)  # 括号中为需要打印的canvas，这里就是在图片上直接打印
    font = ImageFont.truetype("simsun.ttc", 25, encoding="utf-8")  # 第一个参数为字体文件路径，第二个为字体大小
    draw.text((x0, y0), txt, (0, 255, 0), font=font)  # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
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
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
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


    # if the box center locate in fisherROI
    # boxes_center = [(boxes[:,0]+boxes[:,2])/2,(boxes[:,0]+boxes[:,2])/2]
    # result = cv2.pointPolygonTest(biggest, (w, h), False)


    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < min(thresh):
        return im
    # for each detected box
    for i, cls_id in enumerate(classes[0:]):
        # if the box center locate in fisherROI
        box_center = [(boxes[i, 0] + boxes[i, 2]) / 2, (boxes[i, 1] + boxes[i, 3]) / 2]
        result = cv2.pointPolygonTest(fisherROI, (box_center[0], box_center[1]), False)
        if result <=0:
            continue

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
            im = vis_class(im, bbox, class_str, score)
        frame["img"] = im
    return frame

def camera(width, height, fps, args):
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
    # class_names =[
    #     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #     'bus', 'train', 'truck']

    # class_names = [
    #     '__background__', u'人', u'自行车', u'小汽车', u'摩托车', 'airplane',
    #     u'公共汽车', 'train', u'卡车']
    # color_list=[[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,255,0],[255,0,255],[255,255,255]]

    #class_names = [
    #    '__background__', u'人', u'自行车', u'车', u'摩托车', 'airplane',
    #    u'车', 'train', u'车']
    #color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 255], [0, 0, 255],
    #              [255, 0, 255], [0, 0, 255]]
    class_names = ['__background__', u'车']
    color_list = [[0, 0, 0], [0, 255, 0]]

    #cls_sel = [1, 2, 3, 4, 6, 8]
    cls_sel = [1]
    #cls_thresh = [1, 0.5, 0.6, 0.8, 0.6, 0.9, 0.5, 0.9, 0.5]
    cls_thresh = [1, 0.2]
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
    fps = 25
    print("%f fps"%fps)
    print(type(cam))
    #record
    #fourcc = cv2.VideoWriter_fourcc(b'X', b'V', b'I', b'D')
    #fourcc = capture.get(cv2.CAP_PROP_FOURCC)
    fourcc = 1196444237.0
    print(fourcc)
    videoWriter = cv2.VideoWriter('/home/long/objectdetection/vpa_01010002855000000_out.avi', int(fourcc), 3, (int(size[0])/2,int(size[1])/2))
    count2 = 0
    et=0
    # fisherROI = [193,173,342,110,538,97,747,148,797,173,813,290,796,388,670,425,482,439,270,413,177,360]
    while (1):
        frame = {}
        ret, im = capture.read()
        #if type(cam)==str:
        #    im=cv2.resize(im, None, None, fx= width/size[0], fy= height/size[1], interpolation=cv2.INTER_LINEAR)
        ori_im = copy.deepcopy(im)
        frame["img"] = im
        _, contours, hierarchy = cv2.findContours(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im,[fisherROI],-1,(0,255,0),4)
        # # detect one image
        if running == False:
            # if type(cam)==str:
            #     time.sleep(1/fps)
            #count = 0
            count = count+1
            if count%1000:
                cv2.imshow("object detection", frame["img"])

        else:
            count =count+1
            # if count == 2:
            #     start_time = time.time()
            # count = count + 1
            if count%8==1:
                st = time.time()
                with c2_utils.NamedCudaScope(0):
                    cls_boxes, _, _ = infer_engine.im_detect_all(
                        model, im, None, timers=None)
                demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                                               class_names=class_names, color_list=color_list, cls_sel=cls_sel, frame=frame)
                if count2 >= 1:
                    et = et+time.time()-st
                    avg_fps = (count2) / et
                    cv2.putText(frame["img"], '{:s} {:.2f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255),
                                lineType=cv2.LINE_AA)
                count2 = count2 + 1
            # with c2_utils.NamedCudaScope(0):
            #     cls_boxes, _, _ = infer_engine.im_detect_all(
            #         model, im, None, timers=None)
            # demo_vis_one_imageboxes_opencv(im, cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
            #                                class_names=class_names, color_list=color_list, cls_sel=cls_sel, frame=frame)
            # if count>=2:
            #     avg_fps = (count-1) / (time.time() - start_time)
            #     cv2.putText(frame["img"], '{:s} {:.2f}/s'.format('fps', avg_fps), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255),
            #             lineType=cv2.LINE_AA)
                img=cv2.resize(frame["img"],(960,540))
                videoWriter.write(img)  # write frame to video
                # queue.put(frame)
            #new_im = frame["img"]
            #combine=cv2.hconcat([ori_im, new_im])
            #frame["img"] = combine
            # queue.put(frame)
                cv2.imshow("object detection",frame["img"])
    # videoWriter.write(frame["img"])  # write frame to video
if __name__ == "__main__":
    camera(300, 300, 30, parse_args())


