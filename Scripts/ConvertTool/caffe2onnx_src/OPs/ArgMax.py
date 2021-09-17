import caffe2onnx_src.c2oObject as Node
from typing import List
import copy
from caffe2onnx_src.utils import sgs_check
import pdb

def get_argmax_attributes(layer):
    param = layer.argmax_param
    if param.HasField('out_max_val'):
        out_max_val_arg = int(param.out_max_val)
        sgs_check(out_max_val_arg == 0, "only output index")
    if param.HasField('top_k'):
        top_k_arg = int(param.top_k)
        sgs_check(top_k_arg == 1, "only support top 1")
    if param.HasField('axis'):
        axis_arg = int(param.axis)
    keepdims = int(1)
    dict = {
        "axis": axis_arg,
        "keepdims": keepdims,
    }
    return dict

def get_argmax_outshape(layer, input_shape: List) -> List:
    param = layer.argmax_param
    if param.HasField('axis'):
        axis = int(param.axis)
    else:
        axis = 0
    if axis < 0:
        axis = len(input_shape[0]) + axis
    output_shape = copy.deepcopy(input_shape[0])
    output_shape[axis] = 1
    return [output_shape]

def createArgMax(layer, nodename, inname, outname, input_shape):
    attributes = get_argmax_attributes(layer)
    output_shape = get_argmax_outshape(layer, input_shape)
    node = Node.c2oNode(layer, nodename, "ArgMax", inname, outname, input_shape, output_shape, attributes)

    return node
