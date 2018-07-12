#!/home/long/anaconda/bin/python
# -*- coding: utf-8 -*-

# Created by: VPALab Hailong Zhu and Dayu Jia  with PyQt5 UI code generator 5.5.1

#from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
from multiprocessing import Process, Queue, Pipe, Manager
import threading
import time
# import Queue
import argparse
import logging
import math
import os
import cPickle as pickle

q1 = Queue(maxsize=1)
q2 = Queue(maxsize=1)

import types
import copy
from PIL import Image, ImageDraw, ImageFont
from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import utils.c2 as c2_utils
from utils.timer import Timer
from caffe2.python import workspace
import utils.logging
from collections import defaultdict

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
             np.array([1626,  580]), np.array([1592,  776]), np.array([1340,  850]), np.array([964, 878]), np.array([540, 826]),
             np.array([354, 720])]
fisherROI = np.array(fisherROI)


LTcoords = np.array([354,194])
RDcoords = np.array([1626,878])
RoiRect = np.array([1272,684])

reducedROI = fisherROI-LTcoords
# fisherROI = [np.array(i)*2 for i in fisherROI]


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--camera',
        dest='camera',
        help='input option(/path/to/video or camera index )',
        # default='01010002855000000.mp4',
        default='/home/long/objectdetection/01010002855000000.avi',
        type=str
    )

    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/long/objectdetection/res18_1mlp_fpn64.yaml',
        #default='fisher221_1gpu_e2e_faster_rcnn_R-50-FPN.yaml',
        # default = '/home/long/github/detectron/configs/getting_started/fast_rcnn_1mlp_fpn128_res50.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/media/E/models/detectron/res18_1mlp_fpn64_320.pkl',
        # default='/home/long/github/detectron/detectron-output/res18_1mlp_fpn64_512/train/fisher_train_221:fisher_val_221/generalized_rcnn/model_final.pkl',

        #default = '/media/E/models/detectron/compactfishfasterfpn50BN.pkl',
        #default = "model300_final.pkl",
        #default='/home/nvidia/VILab/caffe2detection/compactfishfasterfpn50fp16.pkl',
        #default='/home/nvidia/VILab/caffe2detection/compact1mlpfpn128fp16.pkl',
        # default='/home/long/github/detectron/detectron-output/fish_1mlp_fpn128_512/train/fisher_train_221:fisher_val_221:fisher_valtt_221/generalized_rcnn/model_final.pkl',
        type=str
    )

    return parser.parse_args()


def get_class_string(class_index, class_names):
    class_text = class_names[class_index] if class_names is not None else \
        'id{:d}'.format(class_index)
    return class_text

def vis_class_cn(img, pos, class_str, score):
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

def vis_class(img, pos, class_str, score):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # theta, radius = computeaspect(pos)
    # thetaText = u'%d度'%(90-theta)
    # distText=u'%.2fm'%radius
    # txt = class_str+thetaText+distText
    # txt = class_str+'  '+'%.2f'%score
    txt = '%.2f'%score
    cv2.putText(img, txt, (x0+10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),lineType=cv2.LINE_AA)
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
                                   class_names=[], color_list=[], cls_sel=[]):
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
        result = cv2.pointPolygonTest(reducedROI, (box_center[0], box_center[1]), False)
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
    return im



def postProcessing(timers=None):
    class_names = ['__background__', u'车']
    color_list = [[0, 0, 0], [0, 255, 0]]
    cls_sel = [1]
    cls_thresh = [1, 0.5]
    count1=0
    while(1):
        if not q2.empty():
            timers['vis_detect_bbox'].tic()
            timers['vis_blobs_get'].tic()
            # q2.get()
            # resDict=d2
            resDict =q2.get()
            img = resDict['img'][0]
            rois = resDict['rois']
            scores = resDict['scores']
            box_deltas = resDict['box_deltas']
            im_scales = resDict['im_scales']
            Roi_im_batch = [img[194 - 20:879 + 20, 354 - 5:1627 + 30, :]]
            timers['vis_blobs_get'].toc()
            cls_boxes, _, _ = infer_engine.im_detect_all_batch_multipro_post(Roi_im_batch, rois, scores, box_deltas, im_scales,timers)
            Roi_im=demo_vis_one_imageboxes_opencv(Roi_im_batch[0], cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                                                   class_names=class_names, color_list=color_list, cls_sel=cls_sel)
            # img[194 - 20:879 + 20, 354 - 5:1627 + 30, :] = Roi_im
            cv2.drawContours(img, [fisherROI], -1, (0, 255, 0), 4)
            cv2.imshow('objectDet', img)
            timers['vis_detect_bbox'].toc()

            count1+=1
            if count1==1:
                st=time.time()
            if count1>=2:
                et=time.time()
                acct=et-st
                avgt=acct/(count1-1.0)
                fps=1/avgt
                print('avg fps: %.2f'%fps)
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



def detInference(model=None,timers=None):

    count=0
    with c2_utils.NamedCudaScope(0):
        while (1):

            # # detect one image, batch=1
            # print(d)
            if not q1.empty():
                timers['im_detection'].tic()
                timers['input_blobs_get'].tic()
                # inputDict = copy.deepcopy(d)
                # q1.get()
                inputDict=q1.get()
                Roi_im_batch = [inputDict['img'][0][194 - 20:879 + 20, 354 - 5:1627 + 30, :]]
                timers['input_blobs_get'].toc()
                # print(inputDict)
                timers['im_detecting'].tic()

                # with c2_utils.NamedCudaScope(0):
                    # get_blob_time = time.time()
                    # inputs, im_scales = infer_engine._get_blobs3(fbatch, None)
                    # print("get im blob time %f.2ms" % ((time.time() - get_blob_time) * 1000))
                rois, scores, box_deltas, im_scales= infer_engine.im_detect_all_batch_multipro(model, Roi_im_batch, inputDict['blobs'], inputDict['info'], timers=timers)
                # cls_boxes, _, _ = infer_engine.im_detect_all_batch_multipro_post(rois, scores, box_deltas, im_scales)
                timers['im_detecting'].toc()

                timers['output_blobs_put'].tic()
                if q2.empty():
                    # visDict = {'img':inputDict['img'],'bbox':cls_boxes}
                    # for k, v in visDict.items():
                    #     d2[k] = v
                    # # d2['img']=inputDict['img']
                    # # d2['bbox']=cls_boxes
                    # q2.put(1)
                    # q2.put({'img':inputDict['img'],'bbox':cls_boxes})
                    q2.put({'img':inputDict['img'], 'rois':rois, 'scores':scores, 'box_deltas':box_deltas, 'im_scales':im_scales})
                timers['output_blobs_put'].toc()

                timers['im_detection'].toc()
                for k, v in timers.items():
                    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
                if count == 1:
                    logger.info(
                        ' \ Note: inference on the first image will be slower than the '
                        'rest (caches and auto-tuning need to warm up)'
                    )
            # else:
            #     time.sleep(5)


def preProcessing(width, height, fps, cam, timers=None):

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
    fourcc = 1196444237.0
    count=0

    print(fourcc)
    while (1):
        timers['retrieve_img'].tic()

        ret, im = capture.read()
        fbatch = [im]
        Roi_im = im[194 - 20:879 + 20, 354 - 5:1627 + 30, :]
        # Roi_im = copy.deepcopy(im[194-20:879+20,354-5:1627+30,:])

        inputs, im_scales = infer_engine._get_blobs3([Roi_im], None)

        # if count%8==0 and q.empty():
        timers['input_blobs_put'].tic()

        if q1.empty():
            # inputDict = {'img': fbatch, 'blobs': inputs, 'info': im_scales}
            # for k,v in inputDict.items():
            #     d[k]=v
            # q1.put(1)
            # print(d)
            q1.put({'img': fbatch, 'blobs': inputs, 'info': im_scales})
            # q1.put(pickle.dumps({'img': fbatch, 'blobs': inputs, 'info': im_scales},protocol=-1))
        # input_pipe.send(pickle.dumps({'img': fbatch, 'blobs': inputs, 'info': im_scales},protocol=pickle.HIGHEST_PROTOCOL))
        timers['input_blobs_put'].tic()

        count += 1

        timers['retrieve_img'].toc()
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        time.sleep(1/25.0)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def prepost(width, height, fps, cam, timers=None):
# capture
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    size = (float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(size)

    fps = capture.get(cv2.CAP_PROP_FPS)
    # fps = 25
    print("%f fps" % fps)
    print(type(cam))
    fourcc = 1196444237.0
    count = 0

# visual
    class_names = ['__background__', u'车']
    color_list = [[0, 0, 0], [0, 255, 0]]
    cls_sel = [1]
    cls_thresh = [1, 0.5]
    count1 = 0


    print(fourcc)
    while (1):
        timers['retrieve_img'].tic()

        ret, im = capture.read()
        fbatch = [im]
        Roi_im = im[194 - 20:879 + 20, 354 - 5:1627 + 30, :]
        # Roi_im = copy.deepcopy(im[194-20:879+20,354-5:1627+30,:])

        inputs, im_scales = infer_engine._get_blobs3([Roi_im], None)


        # if count%8==0 and q.empty():
        timers['input_blobs_put'].tic()

        if q1.empty():
            # inputDict = {'img': fbatch, 'blobs': inputs, 'info': im_scales}
            # for k,v in inputDict.items():
            #     d[k]=v
            # q1.put(1)
            # print(d)
            q1.put({'img': fbatch, 'blobs': inputs, 'info': im_scales})
            # q1.put(pickle.dumps({'img': fbatch, 'blobs': inputs, 'info': im_scales},protocol=-1))
        # input_pipe.send(pickle.dumps({'img': fbatch, 'blobs': inputs, 'info': im_scales},protocol=pickle.HIGHEST_PROTOCOL))
        timers['input_blobs_put'].tic()

        count += 1
        timers['retrieve_img'].toc()


        if not q2.empty():
            timers['vis_detect_bbox'].tic()

            timers['vis_blobs_get'].tic()
            # q2.get()
            # resDict=d2
            resDict =q2.get()
            img = resDict['img'][0]
            rois = resDict['rois']
            scores = resDict['scores']
            box_deltas = resDict['box_deltas']
            im_scales = resDict['im_scales']
            Roi_im_batch = [img[194 - 20:879 + 20, 354 - 5:1627 + 30, :]]
            timers['vis_blobs_get'].toc()
            cls_boxes, _, _ = infer_engine.im_detect_all_batch_multipro_post(Roi_im_batch, rois, scores, box_deltas, im_scales,timers)
            Roi_im=demo_vis_one_imageboxes_opencv(Roi_im_batch[0], cls_boxes, thresh=cls_thresh, show_box=True, show_class=True,
                                                   class_names=class_names, color_list=color_list, cls_sel=cls_sel)
            # img[194 - 20:879 + 20, 354 - 5:1627 + 30, :] = Roi_im
            cv2.drawContours(img, [fisherROI], -1, (0, 255, 0), 4)
            cv2.imshow('objectDet', img)
            timers['vis_detect_bbox'].toc()

            count1+=1
            if count1==1:
                st=time.time()
            if count1>=2:
                et=time.time()
                acct=et-st
                avgt=acct/(count1-1.0)
                fps=1/avgt
                print('avg fps: %.2f'%fps)
            for k, v in timers.items():
                logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__=='__main__':
    args=parse_args()
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg()
    timers = defaultdict(Timer)


    cam = args.camera
    if '.' not in cam:
        cam = int(cam)

    manager = Manager()
    d = manager.dict()
    d2 = manager.dict()
    # d = {}
    # 父进程创建Queue，并传给各个子进程：
    # q1 = Queue(maxsize=1)
    # q2 = Queue(maxsize=1)


    # parent_conn, child_conn = Pipe()

    # ppre = Process(target=prepost, args=(300, 300, 30, cam, timers))
    ppre = Process(target=preProcessing, args=(300, 300, 30, cam, timers))
    pdet = threading.Thread(target=detInference, args=(model,timers))

    ppost = threading.Thread(target=postProcessing, args=(timers,))
    # 启动子进程ppre，从摄像头读入图像并准备网络输入:
    # preProcessing(q1, 300, 300, 30, cam)
    # ppre.setDaemon(True)
    ppre.Daemon=True
    pdet.setDaemon(True)
    ppre.start()
    # 启动子进程pdet，执行检测网络:
    pdet.start()
    # # # # 启动子进程post，对输出进行处理和显示:
    ppost.start()
    # # 等待pw结束:
    ppost.join()
    # ppre.join()
    # # pr进程里是死循环，无法等待其结束，只能强行终止:
    # ppre.terminate()
    os._exit(0)



