#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: www.github.com/GustavZ
"""
import numpy as np
import tensorflow as tf
import os
from rod.helper import FPS, WebcamVideoStream, SessionWorker, conv_detect2track, conv_track2detect, vis_detection, Timer
from rod.model import Model
from rod.config import Config
from rod.utils import ops as utils_ops


def detection(model,config):

    print("> Building Graph")
    # tf Session Config
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True
    detection_graph = model.detection_graph
    category_index = model.category_index
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf_config) as sess:
            # start Videostream
            vs = WebcamVideoStream(config.VIDEO_INPUT,config.WIDTH,config.HEIGHT).start()
            # Define Input and Ouput tensors
            tensor_dict = model.get_tensordict(['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks'])
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Mask Transformations


            fps = FPS(config.FPS_INTERVAL).start()
            print('> Starting Detection')
            while vs.isActive():
                # Detection
                if not (config.USE_TRACKER):
                    # default session
                    frame = vs.read()
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: vs.expanded()})
                    num = output_dict['num_detections'][0]
                    classes = output_dict['detection_classes'][0]
                    boxes = output_dict['detection_boxes'][0]
                    scores = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        masks = output_dict['detection_masks'][0]
                    else:
                        masks = None

                    # reformat detection
                    num = int(num)
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.uint8)
                    scores = np.squeeze(scores)

                    # Visualization
                    vis = vis_detection(frame, boxes, classes, scores, masks, category_index, fps.fps_local(),
                                        config.VISUALIZE, config.DET_INTERVAL, config.DET_TH, config.MAX_FRAMES,
                                        fps._glob_numFrames, config.OD_MODEL_NAME)
                    if not vis:
                        break

                fps.update()

    # End everything
    vs.stop()
    fps.stop()

if __name__ == '__main__':
    config = Config()
    model = Model('od',config.OD_MODEL_NAME,config.OD_MODEL_PATH,config.LABEL_PATH,
                config.NUM_CLASSES,config.SPLIT_MODEL, config.SSD_SHAPE).prepare_od_model()
    detection(model, config)
