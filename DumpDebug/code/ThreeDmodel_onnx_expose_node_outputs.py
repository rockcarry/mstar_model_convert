import argparse
import cv2
import onnx
import onnx.shape_inference
import pdb
import numpy as np
import onnxruntime
import copy
import onnx.utils
import sys
import os
import struct
if 'SGS_IPU_DIR' in os.environ:
  Project_path = os.environ['SGS_IPU_DIR']
  sys.path.append(os.path.join(Project_path, 'Scripts/ConvertTool'))
  sys.path.append(os.path.join(Project_path, 'Scripts/calibrator'))
elif 'TOP_DIR' in os.environ:
  Project_path = os.environ['TOP_DIR']
  Project_path1 = os.path.join(Project_path, '../Tool/Scripts/ConvertTool')
  if Project_path1 not in sys.path:
        sys.path.append(Project_path1)
  Project_path2 = os.path.join(Project_path, '../Tool/Scripts/calibrator')
  if Project_path2 not in sys.path:
        sys.path.append(Project_path2)
else:
  raise OSError('Run source cfg_env.sh in top directory.')
from mace.python.tools.sgs_onnx import symbolic_shape_infer
from utils import misc


def expose_node_outputs(model_path, overwrite, run_checker=True, verbose=False):
    """
    Exposes each intermediary node as one of the ONNX model's graph outputs.
     This allows inspection of data passing through the model.
    :param model_path: (str) The path to the .onnx model, with the extension.
    :param overwrite:  (boolean) If true, will overwrite the .onnx file at model_path, else will make a copy.
    :param run_checker: (boolean) If true, will run the ONNX model validity checker
    :param verbose: (boolean) If true, will print detailed messages about execution of preprocessing script
    :return: (str) file path to new ONNX model
    """

    if not overwrite:
        # Make a copy of the .onnx model to save it as a file
        model_path_components = model_path.rsplit(".", 1)  # Split before and after extension
        model_path_new = model_path_components[0] + '_exposed_nodes.' + model_path_components[1]
    # 1. Get name of all external outputs to the model (ie. graph-level outputs, not internal outputs shared bw nodes)
    model = onnx.load(model_path)
    external_outputs = [output.name for output in model.graph.output]
    extended_outputs = []

    # 2. Get the list of nodes in the graph
    for i, node in enumerate(model.graph.node):
        # 3. For every node, copy its (internal) output over to graph.output to make it a graph output
        output_name = [output for output in node.output if output not in external_outputs]
        extended_outputs.extend(output_name)
        for output in output_name:
            intermediate_layer_value_info = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.UNDEFINED, None,
                                                                               'Added to expose Intermediate Node data')
            model.graph.output.extend([intermediate_layer_value_info])

    first_node = copy.deepcopy(model.graph.output[0])
    model.graph.output.remove(model.graph.output[0])
    model.graph.output.extend([first_node])
    if verbose:
        print('The following nodes were exposed as outputs in the {} model:\n {}'.format(model_path, extended_outputs))

    # If all outputs were already "external", no changes are required to the ONNX model, return it as-is
    if len(external_outputs) == len(model.graph.output):
        if verbose:
            print('No change required for ONNX model: All nodes already exposed as outputs')
        return model_path

    # 4. Do a shape and type inference pass on the model to ensure they're defined for graph outputs
    #model = onnx.shape_inference.infer_shapes(model)
    symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model,model_path_new)
    model = onnx.load(model_path_new)
    # 4.5 Remove every output node for which the type or shape could not be inferred
    for i, tensor_valueproto in reversed(list(enumerate(model.graph.output))):
        if not tensor_has_valid_type(tensor_valueproto, verbose):
            del model.graph.output[i]

    if run_checker:
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as v:
            # Ignoring this specific error because the ONNX spec says a missing shape is the legal way to define
            # a tensor of unknown dimensions. Honestly, believe this is a bug in the checker.
            # See https://github.com/onnx/onnx/issues/2492
            if str(v).endswith("Field 'shape' of type is required but missing."):
                if verbose:
                    print("Warning: Ignoring the following error because it is probably an ONNX Checker error: ", v)
            else:
                raise v

    onnx.save(model, model_path_new)
    return model_path_new

def runOnnxRuntime(model,pic,dump,preprocess_func):
       layer_shapes = getAllShape(model)

       session = onnxruntime.InferenceSession(model)

       #value = get_image(pic)
       value = preprocess_func(pic, norm=True)
       value_temp = preprocess_func(pic, norm=True)
       value_temp = value_temp.astype('float32')
       in_name = "./OUTPUT_DATA/" + 'INPUT' + '.txt'
       value_temp = value_temp.transpose(0,3,4,2,1)
       write2txt(np.array(value_temp), in_name)
       inname = [input.name for input in session.get_inputs()][0]
       outname = [output.name for output in session.get_outputs()]
       print("inputs name:",inname,"outputs name:",outname)
       if (dump == 'false'):
           for each_outname in outname:
               output_shape = list(layer_shapes.get(each_outname))
               prediction = session.run([each_outname], {inname:value})
               data = np.array(prediction).reshape(output_shape)

               if NHWC:
                   if len(output_shape) == 4:
                       data = data.transpose(0,2,3,1)
               if "/" in each_outname:
                   each_outname = each_outname.replace("/","_")
                   each_outname = each_outname.replace(":","")
               dump_name = "./OUTPUT_DATA/" + each_outname + str(output_shape)+ '.txt'
               write2txt(np.array(data), dump_name)
       else:
          with open(r"onnx.bin","wb") as f:
              head = "isFloat: 1\n"
              #head_w = bytearray(struct.pack("s", head))
              f.write(head.encode('ascii'))
              i = 0
              for each_outname in outname:
                  output_shape = list(layer_shapes.get(each_outname))
                  prediction = session.run([each_outname], {inname:value})
                  print(each_outname)
                  data = np.array(prediction).reshape(output_shape)

                  if "/" in each_outname:
                      each_outname = each_outname.replace("/","_")
                      each_outname = each_outname.replace(":","")
                  #write head1
                  head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_outname + " bConstant:0\n"
                  f.write(head1.encode('ascii'))
                  #write head2
                  head2 = "op_out[" + str(i) + "] " + each_outname + " = {\n"
                  f.write(head2.encode('ascii'))
                  #write head3
                  head3 = "//buffer data size: " + str(data.size*4) + "\n"
                  f.write(head3.encode('ascii'))

                  if NHWC:
                      if len(output_shape) == 4:
                          data = data.transpose(0,2,3,1)
                  for meta in data.flat:
                      _ubyteArray = bytearray(struct.pack("f", meta))
                      f.write(_ubyteArray)
                  i +=1



def getAllShape(model):

    graph_shapes_dict = {}
    onnx_model = onnx.load(model)
    #onnx.checker.check_model(onnx_model)
    #onnx_model = onnx.utils.polish_model(onnx_model)
    graph = onnx_model.graph

    def extract_value_info(shape_dict, value_info):
        t = tuple([int(dim.dim_value)
                   for dim in value_info.type.tensor_type.shape.dim])
        #pdb.set_trace()
        if t:
            shape_dict[value_info.name] = t

    for vi in graph.value_info:
        extract_value_info(graph_shapes_dict, vi)
    for vi in graph.input:
        extract_value_info(graph_shapes_dict, vi)
    for vi in graph.output:
        extract_value_info(graph_shapes_dict, vi)
    return graph_shapes_dict

def write2txt(data, txt_name):
    with open(txt_name, 'w') as f:
        for num, value in enumerate(list(data.flat)):
            f.write('{:.6f}  '.format(value))
            if (num + 1) % 16 == 0:
                f.write('\n')

def mkdir(path):
  folder = os.path.exists(path)
  if not folder:
    os.makedirs(path)
    print("---  new folder...  ---")
    print("---  OK  ---")

  else:
    print("---  There is this folder!  ---")

def tensor_has_valid_type(tensor_valueproto, verbose):
    """ Ensures ValueProto tensor element type is not UNDEFINED"""
    if tensor_valueproto.type.tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
        if verbose:
            print('Type could not be inferred for the following output, it will be not be exposed:\n',
                  tensor_valueproto)
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', help='Input ONNX model')
    parser.add_argument('--image_path', help='Given a image path to inference')
    parser.add_argument('--dump_bin', default=False, type=str, help='Dump data save to binary. true or false')
    parser.add_argument('-n', '--preprocess', type=str, default='0',
                        help='Name of model to select image preprocess method')
    args = parser.parse_args()

    model_name = args.preprocess
    if model_name != '0':
        preprocess_func = misc.image_preprocess_func(model_name)

    file = './OUTPUT_DATA/'
    mkdir(file)

    expose_model = expose_node_outputs(args.input_model, False)
    #expose_model = args.input_model
    print("Generate Output Data ....")
    runOnnxRuntime(expose_model,args.image_path,args.dump_bin,preprocess_func)
    print("Finish ! ")


if __name__ == '__main__':
    NHWC = True #onnx is NCHW
    main()
