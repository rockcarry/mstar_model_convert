from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite

def BoxDecoder(sgs_builder,model_config,unpack_output_tensors):
    box_num = 3
    side_x = [13, 26]
    side_y = [13, 26]
    ppw = anchor.ones(2535)
    px=[]
    for i in range(2):
        temp = anchor.index_div_linear(1,1,0,box_num ,side_x[i],side_y[i])
        px.extend(temp)
    pph = anchor.ones(2535)
    py=[]
    for i in range(2):
        temp = anchor.index_div_linear(1,1,0,side_x[i]*box_num,side_x[i],1)
        py.extend(temp)
    pw = anchor.ones(2535)
    ph = anchor.ones(2535)

    sx = list(anchor.ns(13*13*3,1.0/13))+list(anchor.ns(26*26*3,1.0/26))
    sy = sx

    biases0= [[81,82],[135,169],[344,319]]
    biases1= [[10,14],[23,27],[37,58]]
    sw =[x[0]/(2*416) for x in biases0 ]*(13*13)  + [x[0]/(2*416) for x in biases1 ]*(26*26)
    sh =[x[1]/(2*416) for x in biases0 ]*(13*13)  + [x[1]/(2*416) for x in biases1 ]*(26*26)

    config = {"shape" :  [1,2535],
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

    reshape0_out_tensors = []
    reshape0_in_tensors = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    reshape0_in_tensors.append(model_config["input"][0])
    reshape_vector=[]
    for value in [1,507,85]:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape0_vector",reshape_vector)
    sgs_builder.buildTensor([3],"reshape0_shape",sgs_builder.getBufferByName("reshape0_vector"),tflite.TensorType.TensorType().INT32)
    reshape0_in_tensors.append("reshape0_shape")
    sgs_builder.buildTensor([1,507,85],"reshape0_tensor")
    reshape0_out_tensors.append("reshape0_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape0",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape0_newshape = sgs_builder.createReshapeOptions([1,507,85])
    sgs_builder.buildOperator("SGS_reshape0",reshape0_in_tensors,reshape0_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape0_newshape)

    reshape1_out_tensors = []
    reshape1_in_tensors = []
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    reshape1_in_tensors.append(model_config["input"][1])
    reshape_vector=[]
    for value in [1,2028,85]:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape1_vector",reshape_vector)
    sgs_builder.buildTensor([3],"reshape1_shape",sgs_builder.getBufferByName("reshape1_vector"),tflite.TensorType.TensorType().INT32)
    reshape1_in_tensors.append("reshape1_shape")
    sgs_builder.buildTensor([1,2028,85],"reshape1_tensor")
    reshape1_out_tensors.append("reshape1_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape1",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape1_newshape = sgs_builder.createReshapeOptions([1,2028,85])
    sgs_builder.buildOperator("SGS_reshape1",reshape1_in_tensors,reshape1_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape0_newshape)

    concat_out_tensors = []
    concat_in_tensors = []
    concat_in_tensors.append("reshape0_tensor")
    concat_in_tensors.append("reshape1_tensor")
    sgs_builder.buildTensor([1,2535,85],"concat_tensor")
    concat_out_tensors.append("concat_tensor")
    sgs_builder.buildOperatorCode("SGS_concat",sgs_builder.BuiltinOperator.CONCATENATION)
    concat_options = sgs_builder.createConcatenationOptions(1,tflite.ActivationFunctionType.ActivationFunctionType().NONE)
    sgs_builder.buildOperator("SGS_concat",concat_in_tensors, concat_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions,concat_options)

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
    sgs_builder.buildOperatorCode("SGS_unpack",sgs_builder.BuiltinOperator.CUSTOM,cus_code)
    sgs_builder.buildOperator("SGS_unpack",concat_out_tensors, unpack_output_tensors,None,None,options)

    BoxDecoder(sgs_builder,model_config,unpack_output_tensors)

    confidence_out_tensors = []
    confidence_in_tensors = []
    confidence_in_tensors.append(unpack_output_tensors[4])
    sgs_builder.buildTensor(model_config["shape"],"confidence_tensor")
    confidence_out_tensors.append("confidence_tensor")
    sgs_builder.buildOperatorCode("SGS_confidence",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_confidence",confidence_in_tensors,confidence_out_tensors)


    score_log_out_tensors = []
    score_log_in_tensors = []
    log_shape = [1,2535]
    score_log_in_tensors.append(unpack_output_tensors[5])
    sgs_builder.buildTensor(log_shape,"score_log_tensor")
    score_log_out_tensors.append("score_log_tensor")
    sgs_builder.buildOperatorCode("SGS_score_log",tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    sgs_builder.buildOperator("SGS_score_log",score_log_in_tensors,score_log_out_tensors)

    score_out_tensors = []
    score_in_tensors = []
    score_in_tensors.append("confidence_tensor")
    score_in_tensors.append("score_log_tensor")
    sgs_builder.buildTensor(log_shape,"score_tensor")
    score_out_tensors.append("score_tensor")
    sgs_builder.buildOperatorCode("SGS_score_mul",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_score_mul",score_in_tensors,score_out_tensors)

    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append("confidence_tensor")
    nms_in_tensors.append("score_tensor")
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
    1.????????????tensor ??? coordinate score confidence class facecoordinate ????????????
    2.nms ???????????? fast nms / yolo_nms
    3.????????????????????????
    4.????????????tensor ???  bboxcount bboxcoordinate score class facecoordinate ????????????
    5.???????????????input coordinate clip ???[0,1]
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
                   (b"nms_score_threshold",0.00499999989,"float"),
                   (b"nms_iou_threshold",0.449999988,"float")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperator("SGS_nms",nms_in_tensors,nms_out_tensors,None,None,options)


    sgs_builder.subgraphs.append( sgs_builder.buildSubGraph(model_config["input"],nms_out_tensors,model_config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,model_config["name"],sgs_builder.buffers)
    file_identifier = b'TFL3'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf

def get_postprocess():
    model_config = {"name":"caffe_yolo_v3_tiny",
          "input" :  ["layer16-conv", "layer23-conv"],
          "input_shape" : [[1,13,13,255],[1,26,26,255]],
          "shape" : [1,2535],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1]]}

    yolov3 = TFLitePostProcess()
    yolov3_buf = buildGraph(yolov3,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(yolov3_buf)
        f.close()
    print("\nWell Done!" + " " + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()

