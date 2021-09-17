import numpy as np
from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite
import pdb

def BoxDecoder(sgs_builder,model_config,unpack_output_tensors):
    if 'TOP_DIR' in os.environ:
        Project_path = os.environ['TOP_DIR']
        prior_bbox = np.load(os.path.join(Project_path, "SRC/Tool/Scripts/postprocess/", "anchors_ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800.npy"))
    elif 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        prior_bbox = np.load(os.path.join(Project_path, "Scripts/postprocess/", "anchors_ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800.npy"))
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

    sx = anchor.ones(822)
    sy = anchor.ones(822)
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
    sgs_builder.setConfig(config)
    sgs_builder.buildBoxDecoding(unpack_output_tensors[0:4])

def buildGraph(sgs_builder,model_config):
    """

    :return:
    """
    unpack_in_tensors = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    unpack_in_tensors.append(model_config["input"][0])
    unpack_output_tensors = []
    for i in range(4):
        sgs_builder.buildTensor(model_config["shape"],"SGS_unpack"+str(i))
        unpack_output_tensors.append("SGS_unpack"+str(i))
    cus_code = 'PostProcess_Unpack'
    cus_options = [(b"x_offset",0,"int"),
                   (b"x_lengh",1,"int"),
                   (b"y_offset",1,"int"),
                   (b"y_lengh",1,"int"),
                   (b"w_offset",2,"int"),
                   (b"w_lengh",1,"int"),
                   (b"h_offset",3,"int"),
                   (b"h_lengh",1,"int"),
                   (b"confidence_offset",0,"int"),
                   (b"confidence_lengh",0,"int"),
                   (b"scores_offset",0,"int"),
                   (b"scores_lengh",0,"int"),
                   (b"max_score",0,"int")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperatorCode("SGS_unpack",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
    sgs_builder.buildOperator("SGS_unpack",unpack_in_tensors,unpack_output_tensors,None,None,options)

    BoxDecoder(sgs_builder,model_config,unpack_output_tensors)

    pp_max_in_tensors = []
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    pp_max_in_tensors.append(model_config["input"][1])
    pp_max_output_tensors = []
    sgs_builder.buildTensor(model_config["shape"],"SGS_PostProcess_Max")
    sgs_builder.buildTensor(model_config["shape"],"SGS_PostProcess_Classes")
    pp_max_output_tensors.append("SGS_PostProcess_Max")
    pp_max_output_tensors.append("SGS_PostProcess_Classes")
    cus_code = 'PostProcess_Max'
    cus_options = [(b"scores_lengh",21,"int"),
                   (b"skip",1,"int")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperatorCode("PostProcess_Max",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
    sgs_builder.buildOperator("PostProcess_Max",pp_max_in_tensors,pp_max_output_tensors,None,None,options)

    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append("SGS_PostProcess_Max")
    nms_in_tensors.append("SGS_PostProcess_Classes")

    sgs_builder.buildTensor(model_config["out_shapes"][0],"detectionBoxes")
    nms_out_tensors.append("detectionBoxes")
    sgs_builder.buildTensor(model_config["out_shapes"][1],"detectionClasses")
    nms_out_tensors.append("detectionClasses")
    sgs_builder.buildTensor(model_config["out_shapes"][2],"detectionScores")
    nms_out_tensors.append("detectionScores")
    sgs_builder.buildTensor(model_config["out_shapes"][3],"numDetections")
    nms_out_tensors.append("numDetections")
    cus_code = 'TFLite_Detection_NMS'
    sgs_builder.buildOperatorCode("SGS_nms",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
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
    cus_options = [(b"input_coordinate_x1",0,"int"),
                   (b"input_coordinate_y1",1,"int"),
                   (b"input_coordinate_x2",2,"int"),
                   (b"input_coordinate_y2",3,"int"),
                   (b"input_class_idx",5,"int"),
                   (b"input_score_idx",4,"int"),
                   (b"input_confidence_idx",-1,"int"),
                   (b"input_facecoordinate_idx",-1,"int"),
                   (b"output_detection_boxes_idx",0,"int"),
                   (b"output_detection_classes_idx",1,"int"),
                   (b"output_detection_scores_idx",2,"int"),
                   (b"output_num_detection_idx",3,"int"),
                   (b"output_detection_boxes_index_idx",-1,"int"),
                   (b"nms",0,"float"),
                   (b"clip",0,"float"),
                   (b"max_detections",100,"int"),
                   (b"max_classes_per_detection",1,"int"),
                   (b"detections_per_class",1,"int"),
                   (b"num_classes",5,"int"),
                   (b"bmax_score",0,"int"),
                   (b"offline",0,"int"),
                   (b"num_classes_with_background",1,"int"),
                   (b"nms_score_threshold",0.01,"float"),
                   (b"nms_iou_threshold",0.45,"float")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperator("SGS_nms",nms_in_tensors,nms_out_tensors,None,None,options)


    sgs_builder.subgraphs.append( sgs_builder.buildSubGraph(model_config["input"],nms_out_tensors,model_config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,model_config["name"],sgs_builder.buffers)
    file_identifier = b'TFL3'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf



def get_postprocess():
    model_config = {"name":"ssd352_288_mobilenetv1_025_6_classes_sigmoid_anchor800",
          "input" : ["402","403"],
          "input_shape" : [[1,822,4],[1,822,6]],
          "shape" : [1,822],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1]]}

    ssd = TFLitePostProcess()
    ssd_buf = buildGraph(ssd, model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(ssd_buf)
        f.close()
    print("\nWell Done! " + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()

