from TFLitePostProcess import *
from anchor_param import *
from third_party import tflite
import math
from mace.python.tools.convert_util import getIPUVersion

def BoxDecoder(sgs_builder,model_config,unpack_output_tensors):
    output_num = 3
    box_num = 3
    side_x = [14, 28, 56]
    side_y = [24, 48, 96]
    ppw = anchor.ones(21168)
    px=[]
    for k in range(output_num):
        for i in range(side_y[k]*side_x[k]):
            temp = i%side_y[k]
            for n in range(box_num):
                px.append(temp)
    pph = anchor.ones(21168)
    py=[]
    for k in range(output_num):
        for i in range(side_y[k]*side_x[k]):
            temp = math.floor(i/side_y[k])
            for n in range(box_num):
                py.append(temp)
    pw = anchor.ones(21168)
    ph = anchor.ones(21168)

    sx = list(anchor.ns(14*24*3,1.0/24.0))+list(anchor.ns(28*48*3,1.0/48.0)) + list(anchor.ns(56*96*3,1.0/96.0))
    sy = list(anchor.ns(14*24*3,1.0/14.0))+list(anchor.ns(28*48*3,1.0/28.0)) + list(anchor.ns(56*96*3,1.0/56.0))

    biases0= [[125.0,334.0],[155.0,219.0],[204.0,362.0]]
    biases1= [[55.0,62.0],[69.0,162.0],[89.0,253.0]]
    biases2= [[15.0,17.0],[29.0,45.0],[43.0,112.0]]

    sw =[x[0]/1536.0 for x in biases0 ]*(14*24)  + [x[0]/1536.0 for x in biases1 ]*(28*48) + [x[0]/1536.0 for x in biases2 ]*(56*96)
    sh =[x[1]/896.0 for x in biases0 ]*(14*24)  + [x[1]/896.0 for x in biases1 ]*(28*48) + [x[1]/896.0 for x in biases2 ]*(56*96)

    config = {"shape" :  [1,21168],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 1.0,
          "h_scale" : 1.0,
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
    #sgs_builder.buildBoxDecoding(unpack_output_tensors[0:4])
    sgs_builder.buildBoxDecoding2(unpack_output_tensors[0:4])


def buildGraph(sgs_builder,model_config):

    reshape0_out_tensors = []
    reshape0_in_tensors = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(model_config["input_shape"][0],model_config["input"][0])
    reshape0_in_tensors.append(model_config["input"][0])
    reshape_vector=[]
    for value in [1,1008,8]:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape0_vector",reshape_vector)
    sgs_builder.buildTensor([3],"reshape0_shape",sgs_builder.getBufferByName("reshape0_vector"),tflite.TensorType.TensorType().INT32)
    reshape0_in_tensors.append("reshape0_shape")
    sgs_builder.buildTensor([1,1008,8],"reshape0_tensor")
    reshape0_out_tensors.append("reshape0_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape0",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape0_newshape = sgs_builder.createReshapeOptions([1,1008,8])
    sgs_builder.buildOperator("SGS_reshape0",reshape0_in_tensors,reshape0_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape0_newshape)

    reshape1_out_tensors = []
    reshape1_in_tensors = []
    sgs_builder.buildTensor(model_config["input_shape"][1],model_config["input"][1])
    reshape1_in_tensors.append(model_config["input"][1])
    reshape_vector=[]
    for value in [1,4032,8]:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape1_vector",reshape_vector)
    sgs_builder.buildTensor([3],"reshape1_shape",sgs_builder.getBufferByName("reshape1_vector"),tflite.TensorType.TensorType().INT32)
    reshape1_in_tensors.append("reshape1_shape")
    sgs_builder.buildTensor([1,4032,8],"reshape1_tensor")
    reshape1_out_tensors.append("reshape1_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape1",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape1_newshape = sgs_builder.createReshapeOptions([1,4032,8])
    sgs_builder.buildOperator("SGS_reshape1",reshape1_in_tensors,reshape1_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape0_newshape)

    reshape2_out_tensors = []
    reshape2_in_tensors = []
    sgs_builder.buildTensor(model_config["input_shape"][2],model_config["input"][2])
    reshape2_in_tensors.append(model_config["input"][2])
    reshape_vector=[]
    for value in [1,16128,8]:
        reshape_vector += bytearray(struct.pack("i", value))
    sgs_builder.buildBuffer("reshape2_vector",reshape_vector)
    sgs_builder.buildTensor([3],"reshape2_shape",sgs_builder.getBufferByName("reshape2_vector"),tflite.TensorType.TensorType().INT32)
    reshape2_in_tensors.append("reshape2_shape")
    sgs_builder.buildTensor([1,16128,8],"reshape2_tensor")
    reshape2_out_tensors.append("reshape2_tensor")
    sgs_builder.buildOperatorCode("SGS_reshape2",tflite.BuiltinOperator.BuiltinOperator().RESHAPE)
    reshape2_newshape = sgs_builder.createReshapeOptions([1,16128,8])
    sgs_builder.buildOperator("SGS_reshape2",reshape2_in_tensors,reshape2_out_tensors,tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,reshape0_newshape)

    concat_out_tensors = []
    concat_in_tensors = []
    concat_in_tensors.append("reshape0_tensor")
    concat_in_tensors.append("reshape1_tensor")
    concat_in_tensors.append("reshape2_tensor")
    sgs_builder.buildTensor([1,21168,8],"concat_tensor")
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
                   (b"scores_lengh",3,"int"),
                   (b"max_score",1,"int")]
    options = sgs_builder.createFlexBuffer(cus_options)
    sgs_builder.buildOperatorCode("SGS_unpack",sgs_builder.BuiltinOperator.CUSTOM,cus_code)
    sgs_builder.buildOperator("SGS_unpack",concat_out_tensors, unpack_output_tensors,None,None,options)

    BoxDecoder(sgs_builder,model_config,unpack_output_tensors)

    score_out_tensors = []
    score_in_tensors = []
    score_in_tensors.append(unpack_output_tensors[4])
    score_in_tensors.append(unpack_output_tensors[5])
    sgs_builder.buildTensor(model_config["shape"],"score_tensor")
    score_out_tensors.append("score_tensor")
    sgs_builder.buildOperatorCode("SGS_score_mul",tflite.BuiltinOperator.BuiltinOperator().MUL)
    sgs_builder.buildOperator("SGS_score_mul",score_in_tensors,score_out_tensors)

    nms_out_tensors = []
    nms_in_tensors = []
    nms_in_tensors.append("x1_tensor")
    nms_in_tensors.append("y1_tensor")
    nms_in_tensors.append("x2_tensor")
    nms_in_tensors.append("y2_tensor")
    nms_in_tensors.append(unpack_output_tensors[4])
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
                   (b"num_classes",3,"int"),
                   (b"bmax_score",1,"int"),
                   (b"offline",0,"int"),
                   (b"num_classes_with_background",1,"int"),
                   (b"nms_score_threshold",0.8,"float"),
                   (b"nms_iou_threshold",0.40,"float")]
    options = sgs_builder.createFlexBuffer(cus_options)
    sgs_builder.buildOperator("SGS_nms",nms_in_tensors,nms_out_tensors,None,None,options)


    sgs_builder.subgraphs.append( sgs_builder.buildSubGraph(model_config["input"],nms_out_tensors,model_config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,model_config["name"],sgs_builder.buffers)
    if getIPUVersion() == 'M6' or getIPUVersion() == 'I6E':
        file_identifier = b'TFL3'
    else:
        file_identifier = b'SIM2'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf

def get_postprocess():
    model_config = {"name":"yolo_v3_7_2",
          "input" :  ["layer83-out","layer95-out","layer107-out"],
          "input_shape" : [[1,14,24,24],[1,28,48,24],[1,56,96,24]],
          "shape" : [1,21168],
          "out_shapes" : [[1,100,4],[1,100],[1,100],[1]]}

    yolov3 = TFLitePostProcess()
    yolov3_buf = buildGraph(yolov3,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(yolov3_buf)
        f.close()
    print("\nWell Done!" + " " + outfilename  + " generated!\n")
    return outfilename

def model_postprocess():
    return get_postprocess()
