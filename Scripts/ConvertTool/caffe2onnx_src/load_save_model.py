from google.protobuf import text_format
from third_party.caffe import caffe_pb2

import onnx
from onnx import utils


def LoadCaffeModel(net_path, model_path):
    # read prototxt
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_pb2.NetParameter()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    return net, model

def LoadOnnxModel(onnx_path):
    onnxmodel = onnx.load(onnx_path)
    return onnxmodel

def SaveOnnxModel(onnx_model, onnx_save_path, need_polish=True):
    opset_import = onnx_model.opset_import
    print("onnx model opset version:\n",opset_import[0])

    try:
        if need_polish:
            polished_model = onnx.utils.polish_model(onnx_model)
            onnx.save_model(polished_model, onnx_save_path)
        else:
            onnx.save_model(onnx_model, onnx_save_path)
        print("Save model at: " + onnx_save_path)
    except Exception as e:
        print("Save model failed !! :", e)
