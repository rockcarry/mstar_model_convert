import math
import os
import pdb
import numpy as np
import sys
import six

from calibrator_custom import sgs_chalk

if 'IPU_TOOL' in os.environ:
    Project_path = os.environ['IPU_TOOL']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
    prior_bbox = np.load(os.path.join(Project_path, "Scripts/postprocess/", "anchors_ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800.npy"))
elif 'SGS_IPU_DIR' in os.environ:
    Project_path = os.environ['SGS_IPU_DIR']
    sys.path.insert(0, os.path.join(Project_path, "Scripts/postprocess"))
    prior_bbox = np.load(os.path.join(Project_path, "Scripts/postprocess/", "anchors_ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800.npy"))
else:
    raise OSError('Run `source cfg_env.sh` in top directory.')

from  anchor_param import *
from third_party import tflite



def buildGraph(model_config):
    """

    :return:
    """

    prior_bbox_ymin = prior_bbox[:, 0]
    prior_bbox_xmin = prior_bbox[:, 1]
    prior_bbox_ymax = prior_bbox[:, 2]
    prior_bbox_xmax = prior_bbox[:, 3]
    # anchors0 = np.array(prior_bbox_xmax) - np.array(prior_bbox_xmin)
    # anchors1 = np.array(prior_bbox_ymax) - np.array(prior_bbox_ymin)
    # anchors2 = (np.array(prior_bbox_xmin) + np.array(prior_bbox_xmax)) / 2.
    # anchors3 = (np.array(prior_bbox_ymin) + np.array(prior_bbox_ymax)) / 2.

    ppw = prior_bbox_ymax
    px = prior_bbox_ymin
    pph = prior_bbox_xmax
    py = prior_bbox_xmin
    pw = [0.5]
    ph = [0.5]

    sx = np.ones(822)
    sy = np.ones(822)
    sw = prior_bbox_ymax
    sh = prior_bbox_xmax

    config = {"shape" : [1,822],
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

    convert_scores = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])

    postprocess_max_output_list = sgs_chalk.PostProcess_Max(convert_scores,num_classes=21,skip=1)
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
                                          mode='SSD',max_detections=100,nms_score_threshold=0.01,
                                          nms_iou_threshold=0.45,num_classes=5,is_need_index=False)
    """=========================================="""

    outfilename = model_config["name"] + "_postprocess.sim"
    model = sgs_chalk.Model([in0,in1],out1,model_config["name"])
    model.save(outfilename)

    print("\nWell Done! " + outfilename  + " generated!\n")
    return outfilename


def get_postprocess():
    model_config = {"name":"ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800",
          "input" : ["402","403"],
          "input_shape" : [[1,822,4],[1,822,6]],
          "shape" : [1,822],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1]]}

    outfilename = buildGraph(model_config)

    return outfilename

def model_postprocess():
    return get_postprocess()
