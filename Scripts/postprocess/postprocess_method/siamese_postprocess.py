from TFLitePostProcess import *
from  anchor_param import *
from third_party import tflite
from itertools import product as product
from math import ceil
import pdb

def buildGraph(sgs_builder,config):
    """

    :return:
    """
    """=========================================="""

    biases = [-2.1483638286590576]
    biases_vector=[]
    biases_vector += bytearray(struct.pack("f", biases[0]))
    sgs_builder.buildBuffer("biases_vector", biases_vector)
    sgs_builder.buildTensor([1],"biases_vector",sgs_builder.getBufferByName("biases_vector"))

    cc_in_tensors = []
    sgs_builder.buildBuffer('NULL')
    sgs_builder.buildTensor(config["input_shape"][1],config["input"][1])
    cc_in_tensors.append(config["input"][1])
    sgs_builder.buildTensor(config["input_shape"][0],config["input"][0])
    cc_in_tensors.append(config["input"][0])
    cc_in_tensors.append("biases_vector")

    cc_out_tensors = []
    sgs_builder.buildTensor(config["out_shapes"][0],"cross_correlation")
    cc_out_tensors.append("cross_correlation")
    #sgs_builder.buildOperatorCode("SGS_cross_correlation",tflite.BuiltinOperator.BuiltinOperator().CONV_2D)
    #Conv2DOptions = sgs_builder.createConv2DOptions(0,0,0,0,0,0,0,0,0,0)
    #sgs_builder.buildOperator("SGS_cross_correlation",conv_in_tensors,conv_out_tensors,tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions,Conv2DOptions)
    cus_code = 'cross_correlation'
    sgs_builder.buildOperatorCode("cross_correlation",tflite.BuiltinOperator.BuiltinOperator().CUSTOM,cus_code)
    cus_options = [(b"bias",-2.1483638286590576,"float")]
    options = sgs_builder.createFlexBuffer( sgs_builder.lib, cus_options)
    sgs_builder.buildOperator("cross_correlation",cc_in_tensors,cc_out_tensors,None,None,options)

    sgs_builder.subgraphs.append( sgs_builder.buildSubGraph(config["input"],cc_out_tensors,config["name"]))
    sgs_builder.model = sgs_builder.createModel(3,sgs_builder.operator_codes,sgs_builder.subgraphs,config["name"],sgs_builder.buffers)
    file_identifier = b'TFL3'
    sgs_builder.builder.Finish(sgs_builder.model, file_identifier)
    buf = sgs_builder.builder.Output()
    return buf

def get_postprocess():
    model_config = {"name":"siamese",
          "input" : ['inference/convolutional_alexnet/conv5/concat','inference/convolutional_alexnet_1/conv5/concat'],
          "input_shape" : [[1,6,6,256],[1,22,22,256]],
          "shape" : [1,3405],
          "out_shapes" : [[1,17,17,1]],
          "input_hw": [180., 320.]}

    siamese = TFLitePostProcess()
    siamese_buf = buildGraph(siamese,model_config)
    outfilename = model_config["name"] + "_postprocess.sim"
    with open(outfilename, 'wb') as f:
        f.write(siamese_buf)
        f.close()
    print("\nWell Done!" + outfilename  + " generated!\n")

def model_postprocess():
    return get_postprocess()
