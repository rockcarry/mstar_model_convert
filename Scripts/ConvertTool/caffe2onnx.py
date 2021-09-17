import argparse
import sys
import os
import six


from caffe2onnx_src.load_save_model import LoadCaffeModel, SaveOnnxModel
from caffe2onnx_src.mapCaffe2Onnx import Caffe2Onnx
from caffe2onnx_src.args_parser import parse_args
from caffe2onnx_src.utils import is_ssd_model

FLAGS = None

def main(unused_args):
    if not os.path.exists(FLAGS.model_file):
        six.print_("Input graph file '" +
                   FLAGS.model_file +
                   "' does not exist!", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(FLAGS.weight_file):
        six.print_("Input weight file '" + FLAGS.weight_file +
                   "' does not exist!", file=sys.stderr)
        sys.exit(1)

    caffe_graph_path = FLAGS.model_file
    caffe_params_path = FLAGS.weight_file

    pos_s = caffe_graph_path.rfind("/")
    if  pos_s == -1:
        pos_s = 0

    pos_dot = caffe_graph_path.rfind(".")
    onnx_name = caffe_graph_path[pos_s+1:pos_dot]
    save_path = caffe_graph_path[0:pos_dot] + '.onnx'
    if FLAGS.output_dir is not None:
        save_path = FLAGS.output_dir

    graph, params = LoadCaffeModel(caffe_graph_path,caffe_params_path)
    print('2. Begin to Convert')
    c2o = Caffe2Onnx(graph, params, onnx_name)
    print('3. Creat Onnx Model')
    onnx_model = c2o.createOnnxModel()
    print('4. Save Onnx Model')
    is_ssd = is_ssd_model(caffe_graph_path)
    opset_import = onnx_model.opset_import
    if is_ssd:
        SaveOnnxModel(onnx_model, save_path, need_polish=False)
    else:
        SaveOnnxModel(onnx_model, save_path, need_polish=True)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='caffe Convert To Onnx'
    )
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_file", required=True, type=str, default="", help="Caffe prototxt file to load.")
    parser.add_argument(
        "--weight_file", required=True, type=str, default="", help="Caffe data file to load.")
    parser.add_argument(
        "--output_dir", type=str, default="./Converted_Net_float.sim", help="File to save the output onnx model to.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)

