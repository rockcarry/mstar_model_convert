from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite

def BoxDecoder(sgs_builder,model_config,unpack_output_tensors):
    box_num = 5
    side_x = 19
    side_y = 19
    ppw = anchor.ones(1805)
    px = anchor.index_div_linear(1,1,0,box_num ,side_x,side_y)
    pph = anchor.ones(1805)
    py = anchor.index_div_linear(1,1,0,side_x*box_num,side_y,1)
    pw = anchor.ones(1805)
    ph = anchor.ones(1805)

    sx = anchor.ns(1805,1.0/19)
    sy = anchor.ns(1805,1.0/19)

    biases= [[0.57273, 0.677385],[1.87446, 2.06253],[3.33843, 5.47434],[7.88282, 3.52778],[9.77052, 9.16828]]
    sw = [x[0]/(2*19) for x in biases ]*(19*19)
    sh = [x[1]/(2*19) for x in biases ]*(19*19)

    config = {"shape" : [1,1805],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 1,
          "h_scale" : 1,
          "anchor_selector" : "constant",
          "pw" : pw,
          "ph" : ph,
          "pw_func" : (None,None),
          "ph_func" : (None,None),
          "ppw" : ppw,
          "px" : px,
          "pph" : pph,
          "py" : py,
          "sx" : sx,
          "sy" : sy,
          "sw" : sw,
          "sh" : sh
          }
    sgs_builder.setConfig(config)
    sgs_builder.buildBoxDecoding(unpack_output_tensors[0:4])

def buildGraph(sgs_builder,model_config):
    """

    :return:
    """
    reshape_out_shape =  [1,1805,85]
    reshape_out_tensors = []
    reshape_in_tensors = []

    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"],model_config["input"][0])
    reshape_in_tensors.append(model_config["input"][0])
    reshape_vector=[]
    for value in reshape_out_shape:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape_vector",reshape_vector)
    sgs_builder.buildTensor([len(reshape_out_shape)],"reshape_shape",sgs_builder.getBufferByName("reshape_vector"),tflite.TensorType.TensorType().INT32)
    reshape_in_tensors.append("reshape_shape")
    sgs_builder.buildTensor(reshape_out_shape,"reshape_tensor")
    reshape_out_tensors.append("reshape_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape_newshape = sgs_builder.createReshapeOptions(reshape_out_shape)
    sgs_builder.buildOperator("SGS_reshape",reshape_in_tensors, reshape_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape_newshape)

    unpack_output_tensors = []
    for i in range(7):
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
                   (b"confidence_offset",4,"int"),
                   (b"confidence_lengh",1,"int"),
                   (b"scores_offset",5,"int"),
                   (b"scores_lengh",80,"int"),
                   (b"max_score",1,"int")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperatorCode("SGS_unpack",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
    sgs_builder.buildOperator("SGS_unpack",reshape_out_tensors, unpack_output_tensors,None,None,options)

    BoxDecoder(sgs_builder,model_config,unpack_output_tensors)

    confidence_out_tensors = []
    confidence_in_tensors = []
    confidence_in_tensors.append(unpack_output_tensors[4])
    sgs_builder.buildTensor(model_config["shape"],"confidence_tensor")
    confidence_out_tensors.append("confidence_tensor")
    sgs_builder.buildOperatorCode("SGS_confidence",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_confidence",confidence_in_tensors,confidence_out_tensors)


    score0_out_tensors = []
    score0_in_tensors = []
    score0_in_tensors.append(unpack_output_tensors[5])
    sgs_builder.buildTensor(model_config["shape"],"score0_tensor")
    score0_out_tensors.append("score0_tensor")
    sgs_builder.buildOperatorCode("SGS_score0",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_score0",score0_in_tensors,score0_out_tensors)

    score1_out_tensors = []
    score1_in_tensors = []
    score1_in_tensors.append("confidence_tensor")
    score1_in_tensors.append("score0_tensor")
    sgs_builder.buildTensor(model_config["shape"],"SGS_score1")
    score1_out_tensors.append("SGS_score1")
    sgs_builder.buildOperatorCode("SGS_score_mul",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_score_mul",score1_in_tensors,score1_out_tensors)


    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append("confidence_tensor")
    nms_in_tensors.append("SGS_score1")
    nms_in_tensors.append(unpack_output_tensors[6])

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
                   (b"input_class_idx",6,"int"),
                   (b"input_score_idx",5,"int"),
                   (b"input_confidence_idx",4,"int"),
                   (b"input_facecoordinate_idx",-1,"int"),
                   (b"output_detection_boxes_idx",0,"int"),
                   (b"output_detection_classes_idx",1,"int"),
                   (b"output_detection_scores_idx",2,"int"),
                   (b"output_num_detection_idx",3,"int"),
                   (b"output_detection_boxes_index_idx",-1,"int"),
                   (b"nms",0,"float"),
                   (b"clip",1,"float"),
                   (b"max_detections",100,"int"),
                   (b"max_classes_per_detection",1,"int"),
                   (b"detections_per_class",1,"int"),
                   (b"num_classes",80,"int"),
                   (b"bmax_score",1,"int"),
                   (b"offline",0,"int"),
                   (b"num_classes_with_background",1,"int"),
                   (b"nms_score_threshold",0.4,"float"),
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
    model_config = {"name":"caffe_yolo_v2_608",
          "input" : ['layer31-conv'],
          "input_shape" : [1,19,19,425],
          "shape" : [1,1805],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1]]}

    yolov2 = TFLitePostProcess()
    yolov2_buf = buildGraph(yolov2,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(yolov2_buf)
        f.close()
    print("\nWell Done!" + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()

