import math
import os
import pdb
import numpy as np
import sys
import six

from calibrator_custom import sgs_chalk
from third_party import tflite

def anchors_generate():
    anchors = []
    img_width = np.ones(6)*300
    img_height = np.ones(6)*300
    layer_width = [19,10,5,3,2,1]
    layer_height = [19,10,5,3,2,1]
    num_priors = [3,6,6,6,6,6]
    offset_ = 0.5
    min_sizes_ = [[60],[105],[150],[195],[240],[285]]
    max_sizes_ = [[0],[150],[195],[240],[285],[300]]
    aspect_ratios_ = [[1,2,0.5],[1,2,0.5,3,0.33333],[1,2,0.5,3,0.33333],[1,2,0.5,3,0.33333],[1,2,0.5,3,0.33333],[1,2,0.5,3,0.33333]]
    clip_ = False
    step_w = img_width/layer_width
    step_h = img_height/layer_height

    for num in range(len(layer_width)):
        dim = layer_height[num]*layer_width[num]*num_priors[num]*4
        top_data = []
        for h in range(layer_height[num]):
            for w in range(layer_width[num]):
                center_x = (w+offset_)*step_w[num]
                center_y = (h+offset_)*step_h[num]
                for s in range(len(min_sizes_[num])):
                    min_size = min_sizes_[num][s]
                    box_width = min_size
                    box_height = min_size
                    # xmin, ymin, xmax, ymax
                    top_data.append((center_x - box_width/2.) / img_width[s])
                    top_data.append((center_y - box_height/2.) / img_height[s])
                    top_data.append((center_x + box_width/2.) / img_width[s])
                    top_data.append((center_y + box_height/2.) / img_height[s])
                    if max_sizes_[num][s] > 0:
                        max_size = max_sizes_[num][s]
                        box_width = math.sqrt(min_size * max_size)
                        box_height = math.sqrt(min_size * max_size)
                        # xmin, ymin, xmax, ymax
                        top_data.append((center_x - box_width/2.) / img_width[s])
                        top_data.append((center_y - box_height/2.) / img_height[s])
                        top_data.append((center_x + box_width/2.) / img_width[s])
                        top_data.append((center_y + box_height/2.) / img_height[s])
                    for r in range(len(aspect_ratios_[num])):
                        ar = aspect_ratios_[num][r]
                        if (math.fabs(ar - 1.) > 1e-6):
                            box_width = min_size * math.sqrt(ar)
                            box_height = min_size / math.sqrt(ar)
                            # xmin, ymin, xmax, ymax
                            top_data.append((center_x - box_width/2.) / img_width[s])
                            top_data.append((center_y - box_height/2.) / img_height[s])
                            top_data.append((center_x + box_width/2.) / img_width[s])
                            top_data.append((center_y + box_height/2.) / img_height[s])
        if clip_ == True:
            for d in dim:
                top_data[d] = math.min(math.max(top_data[d], 0.), 1.)
        anchors = anchors + top_data
    anchors_array = np.array(anchors)
    return anchors_array

def buildGraph(model_config):
    """

    :return:
    """
    mbox_priorbox = sgs_chalk.Input(model_config["input_shape"][2], name = model_config["input"][2])
    anchors = anchors_generate()
    prior_bbox = anchors.reshape((1917,4))
    prior_bbox_xmin = [x[0] for x in prior_bbox]
    prior_bbox_ymin = [x[1] for x in prior_bbox]
    prior_bbox_xmax = [x[2] for x in prior_bbox]
    prior_bbox_ymax = [x[3] for x in prior_bbox]
    anchors0 = np.array(prior_bbox_xmax) - np.array(prior_bbox_xmin)
    anchors1 = np.array(prior_bbox_ymax) - np.array(prior_bbox_ymin)
    anchors2 = (np.array(prior_bbox_xmin) + np.array(prior_bbox_xmax)) / 2.
    anchors3 = (np.array(prior_bbox_ymin) + np.array(prior_bbox_ymax)) / 2.

    ppw = anchors0
    px = anchors2
    pph = anchors1
    py = anchors3
    pw = anchors0
    ph = anchors1

    sx = np.ones(1917)
    sy = np.ones(1917)
    sw = np.ones(1917)*0.5
    sh = np.ones(1917)*0.5

    config = {"shape" : [1,1917],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"x_scale"),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"y_scale"),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"w_scale"),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().MUL,"h_scale"),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 0.2,
          "h_scale" : 0.2,
          "anchor_selector" : "constant",
          "pw" : pw,
          "ph" : ph,
          "pw_func" : (tflite.BuiltinOperator.BuiltinOperator().SQUARE,None),
          "ph_func" : (tflite.BuiltinOperator.BuiltinOperator().SQUARE,None),
          "ppw" : ppw,
          "px" : px,
          "pph" : pph,
          "py" : py,
          "sx" : sx,
          "sy" : sy,
          "sw" : sw,
          "sh" : sh,
          }

    in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])

    unpack_out_tensors1 = []
    for i in range(4):
        unpack_out_tensors1.append("SGS_unpack"+str(i))
    output_list = sgs_chalk.PostProcess_Unpack(in0,mode='SSD',name=unpack_out_tensors1)

    bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
    mbox_conf_softmax = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])

    postprocess_max_output_list = sgs_chalk.PostProcess_Max(mbox_conf_softmax,num_classes=21,is_skip_background=1)
    """=========================================="""
    '''
    1.标识输入tensor 和 coordinate score confidence class facecoordinate 对应关系
    2.nms 计算方式 fast nms / yolo_nms
    3.背景是否参与运算
    4.标识输出tensor 和  bboxcount bboxcoordinate score class facecoordinate 对应关系
    5.是否需要将input coordinate clip 到[0,1]
                       (b"input_class_idx",4,"int"),
                   (b"input_score_idx",5,"int"),
                   (b"input_confidence_idx",6,"int"),
    '''
    out1 = sgs_chalk.TFLite_Detection_NMS(bosdecoder_output_list[0],bosdecoder_output_list[1],bosdecoder_output_list[2],
                                          bosdecoder_output_list[3],postprocess_max_output_list[0],postprocess_max_output_list[1],
                                          mode='SSD',max_detections=10,nms_score_threshold=0.01,
                                          nms_iou_threshold=0.45,num_classes=20,clip=0,is_need_index=False)

    """=========================================="""

    outfilename = model_config["name"] + "_postprocess.sim"
    model = sgs_chalk.Model([in0,mbox_conf_softmax,mbox_priorbox],out1,model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename


def get_postprocess():
    model_config = {"name":"caffe_ssd_mobilenet_v1",
          "input" : ["mbox_loc","mbox_conf_softmax","mbox_priorbox"],
          "input_shape" : [[1,1917,4],[1,1917,21],[1917,4]],
          "shape" : [1,1917],
          "out_shapes" : [[1,10,4],[1,10],[1,10],[1]]}

    outfilename = buildGraph(model_config)

    return outfilename

def model_postprocess():
    return get_postprocess()
