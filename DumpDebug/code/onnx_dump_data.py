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
import importlib
from calibrator_custom import utils
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
from onnxruntime.tools import symbolic_shape_infer
from utils import misc

def image_preprocess_func(model_name):
    if os.path.exists(model_name) and model_name.split('.')[-1] == 'py':
        sys.path.append(os.path.dirname(model_name))
        preprocess_func = importlib.import_module(os.path.basename(model_name).split('.')[0])
    else:
        raise Exception('No preprocess function found in {}'.format(model_name))
    return preprocess_func.image_preprocess

def image_generator(image_list, preprocess_funcs, norm=True):
    for image in image_list:
        imgs = []
        if len(preprocess_funcs) > 1:
            if len(preprocess_funcs) != len(image):
                raise ValueError('Got different num of preprocess_methods and images!')
            for idx, preprocess_func in enumerate(preprocess_funcs):
                pre_img = preprocess_func(image[idx], norm)
                imgs.append(pre_img.transpose(0, 3, 1, 2) if len(pre_img.shape) == 4 else pre_img)
            yield [imgs]
        else:
            if isinstance(image, list):
                if len(preprocess_funcs) != 1 or len(image) != 1:
                    raise ValueError('Got wrong num of preprocess_methods and images!')
                pre_img = preprocess_funcs[0](image[0], norm)
                imgs.append(pre_img.transpose(0, 3, 1, 2) if len(pre_img.shape) == 4 else pre_img)
                yield [imgs]
            else:
                pre_img = preprocess_funcs[0](image, norm)
                imgs.append(pre_img.transpose(0, 3, 1, 2) if len(pre_img.shape) == 4 else pre_img)
                yield [imgs]

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.basename(apath).split('.')[-1].lower() in image_suffix:
                result.append(os.path.abspath(apath))
    if len(result) == 0:
        raise FileNotFoundError('No images found in {}'.format(dirname))
    return result

image_suffix = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                'sr', 'ras', 'tiff', 'tif', 'hdr', 'pic', 'raw', 'npy', 'bin', 'mp4']

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
    model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(model)
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

def convert_to_ndtype(tensor_type):
    if tensor_type == 'tensor(double)':
        return np.float64
    elif tensor_type == 'tensor(float)':
        return np.float32
    elif tensor_type == 'tensor(int32)':
        return np.int32
    else:
        raise ValueError('Unknown type: %s' % tensor_type)

def runOnnxRuntime(model_path, images, dump_bin, preprocess_funcs):
        layer_shapes = getAllShape(model_path)

        session = onnxruntime.InferenceSession(model_path)
        #pdb.set_trace()
        #value = get_image(pic)

        model = onnx.load(model_path)
        inputs = [m.name for m in model.graph.input]
        inputs_type = [convert_to_ndtype(m.type) for m in session.get_inputs()]
        inname = [input.name for input in session.get_inputs()]
        outname = [output.name for output in session.get_outputs()]

        if len(inname) != len(images) or len(inname) != len(preprocess_funcs):
            raise ValueError('Got different num of inputs and images!')

        output_img = []
        for idx, preprocess in enumerate(preprocess_funcs):
            pre_img = preprocess(images[idx], norm=True)
            output_img.append(pre_img.transpose(0, 3, 1, 2) if len(pre_img.shape) == 4 else pre_img)

        input_feed = [(inname[idx], output_img[idx].astype(inputs_type[idx])) for idx, _ in enumerate(inname)]
        input_feed = dict(input_feed)

        print("inputs name:",inname,"outputs name:",outname)

        if (dump_bin == "False"):
            file_total = open("dumpData/onnx_NHWC_outtensor_dump.txt",'w')
            head = "isFloat: 1\n"
            file_total.write(head)
        else:
            file_total = open("dumpData/onnx_NHWC_outtensor_dump.bin",'wb')
            head = "isFloat: 1\n"
            file_total.write(head.encode('ascii'))

        if (dump_bin == "False"):
            #dump input tensors
            i = 0
            for each_inname in inname:
                raw_data = np.array(input_feed[each_inname])
                if len(raw_data.shape) == 4:
                    raw_data_NHWC = np.transpose(raw_data,[0,2,3,1])
                else:
                    raw_data_NHWC = raw_data

                each_inname = each_inname.replace("/","_")
                each_inname = each_inname.replace(":","_")
                each_inname = each_inname.replace("*","_")
                each_inname = each_inname.replace("?","_")
                each_inname = each_inname.replace("<","_")
                each_inname = each_inname.replace(">","_")
                each_inname = each_inname.replace("|","_")

                # dump each input in seperated files as NCHW string
                f = open('dumpData/NCHW/'+each_inname + '#' + str(raw_data.shape),'w')
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_inname + " bConstant:0 " + "shape:["
                for j in range(len(raw_data.shape)):
                    if j == (len(raw_data.shape) - 1):
                        head1 = head1 + str(raw_data.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
                f.write(head1)
                head2 = "input[" + str(i) + "] " + each_inname + " = {\n"
                f.write(head2)
                head3 = "//buffer data size: " + str(raw_data.size*4) + "\n\n"
                f.write(head3)

                for num, value in enumerate(list(raw_data.flat)):
                    f.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('}\n')
                f.close

                # dump each input in seperated files as NHWC string
                f = open('dumpData/NHWC/'+each_inname + '#' + str(raw_data_NHWC.shape),'w')
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_inname + " bConstant:0 " + "shape:["
                for j in range(len(raw_data_NHWC.shape)):
                    if j == (len(raw_data_NHWC.shape) - 1):
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data_NHWC.shape)) + "\n"
                f.write(head1)
                head2 = "input[" + str(i) + "] " + each_inname + " = {\n"
                f.write(head2)
                head3 = "//buffer data size: " + str(raw_data_NHWC.size*4) + "\n\n"
                f.write(head3)


                for num, value in enumerate(list(raw_data_NHWC.flat)):
                    f.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('}\n')
                f.close
                i +=1

            #dump output tensors
            i = 0
            for each_outname in outname:
                output_shape = list(layer_shapes.get(each_outname))
                prediction = session.run([each_outname], input_feed)
                raw_data = np.array(prediction).reshape(output_shape)
                node_name = ''
                for node in model.graph.node:
                    if each_outname in node.output:
                        if node.name != '':
                            node_name = node.name
                        else:
                            node_name = node.op_type
                        break

                if len(raw_data.shape) == 4:
                    raw_data_NHWC = np.transpose(raw_data,[0,2,3,1])
                else:
                    raw_data_NHWC = raw_data

                each_outname_file = each_outname.replace("/","_")
                each_outname_file = each_outname_file.replace(":","_")
                each_outname_file = each_outname_file.replace("*","_")
                each_outname_file = each_outname_file.replace("?","_")
                each_outname_file = each_outname_file.replace("<","_")
                each_outname_file = each_outname_file.replace(">","_")
                each_outname_file = each_outname_file.replace("|","_")

                # dump each layer in seperated files as NCHW string
                f = open('dumpData/NCHW/'+each_outname_file + '#' + str(raw_data.shape),'w')
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_outname + " bConstant:0 " + "shape:["
                for j in range(len(raw_data.shape)):
                    if j == (len(raw_data.shape) - 1):
                        head1 = head1 + str(raw_data.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
                f.write(head1)
                head2 = "op_out[" + str(i) + "] " + node_name + " = {\n"
                f.write(head2)
                head3 = "//buffer data size: " + str(raw_data.size*4) + "\n\n"
                f.write(head3)

                for num, value in enumerate(list(raw_data.flat)):
                    f.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('}\n')
                f.close

                # dump each layer in seperated files as NHWC string
                f = open('dumpData/NHWC/'+each_outname_file + '#' + str(raw_data_NHWC.shape),'w')
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_outname + " bConstant:0 " + "shape:["
                for j in range(len(raw_data_NHWC.shape)):
                    if j == (len(raw_data_NHWC.shape) - 1):
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data_NHWC.shape)) + "\n"
                f.write(head1)
                head2 = "op_out[" + str(i) + "] " + node_name + " = {\n"
                f.write(head2)
                head3 = "//buffer data size: " + str(raw_data_NHWC.size*4) + "\n\n"
                f.write(head3)


                for num, value in enumerate(list(raw_data_NHWC.flat)):
                    f.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('}\n')
                f.close

                # dump all layers in one file as NHWC string
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_outname + " bConstant:0 " + "shape:["
                for j in range(len(raw_data_NHWC.shape)):
                    if j == (len(raw_data_NHWC.shape) - 1):
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data_NHWC.shape)) + "\n"
                file_total.write(head1)
                head2 = "op_out[" + str(i) + "] " + node_name + " = {\n"
                file_total.write(head2)
                head3 = "//buffer data size: " + str(raw_data_NHWC.size*4) + "\n\n"
                file_total.write(head3)

                for num, value in enumerate(list(raw_data_NHWC.flat)):
                    file_total.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        file_total.write('\n')
                file_total.write('\n')
                file_total.write('}\n')
                i +=1
        else:
            i = 0
            for each_outname in outname:
                layer = layer_shapes.get(each_outname)
                if layer is None:
                    continue
                output_shape = list(layer)
                prediction = session.run([each_outname], input_feed)
                raw_data = np.array(prediction).reshape(output_shape)
                node_name = ''
                for node in model.graph.node:
                    if each_outname in node.output:
                        if node.name != '':
                            node_name = node.name
                        else:
                            node_name = node.op_type
                        break

                if len(raw_data.shape) == 4:
                    raw_data_NHWC = np.transpose(raw_data,[0,2,3,1])
                else:
                    raw_data_NHWC = raw_data

                # dump all layers in one file as NHWC bin
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + each_outname + " bConstant:0 " + "shape:["

                for j in range(len(raw_data_NHWC.shape)):
                    if j == (len(raw_data_NHWC.shape) - 1):
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data_NHWC.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data_NHWC.shape)) + "\n"
                file_total.write(head1.encode('ascii'))
                head2 = "op_out[" + str(i) + "] " + node_name + " = {\n"
                file_total.write(head2.encode('ascii'))
                head3 = "//buffer data size: " + str(raw_data_NHWC.size*4) + "\n"
                file_total.write(head3.encode('ascii'))

                for meta in raw_data_NHWC.flat:
                    _ubyteArray = bytearray(struct.pack("f", meta))
                    file_total.write(_ubyteArray)
                i +=1
        file_total.close


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
    parser.add_argument('--model_file', help='onnx model file')
    parser.add_argument('-i', '--image', help='input images')
    parser.add_argument('--dump_bin', default=False, type=str, help='Dump data save as binary (or string). True or False')
    parser.add_argument('-n', '--preprocess', type=str, default='0',
                    help='Name of model to select image preprocess method')
    args = parser.parse_args()

    model_name = args.preprocess
    image_path = args.image
    dump_bin = args.dump_bin
    norm = True

    preprocess_funcs = [image_preprocess_func(n) for n in model_name.split(',')]

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise FileNotFoundError('No such {} image or directory.'.format(base_name))

    if os.path.isdir(base_name):
        image_list = all_path(base_name)
        img_gen = image_generator(image_list, preprocess_funcs, norm)
    elif os.path.basename(base_name).split('.')[-1].lower() in image_suffix:
        img_gen = image_generator([base_name], preprocess_funcs, norm)
        image_list = [base_name]
    else:
        with open(base_name, 'r') as f:
            multi_images = f.readlines()
        if dir_name is None:
            multi_images = [images.strip().split(',') for images in multi_images]
        else:
            multi_images = [[os.path.join(dir_name, i) for i in images.strip().split(',')] for images in multi_images]
        img_gen = image_generator(multi_images, preprocess_funcs, norm)
        image_list = multi_images

    if len(preprocess_funcs) > 1:
        images = image_list[0]
        if len(image_list[0]) != len(preprocess_funcs):
            raise ValueError('Got different num of preprocess_methods and images!')
    else:
        if isinstance(image_list[0], list):
            images = image_list[0]
            if len(preprocess_funcs) != 1 or len(image_list[0]) != 1:
                raise ValueError('Got wrong num of preprocess_methods and images!')
        else:
            images = [image_list[0]]

    #extract_feature(net, images, preprocess_funcs, dump_bin)

    file = './dumpData/'
    mkdir(file)
    if (dump_bin == "False"):
        file = './dumpData/NHWC/'
        mkdir(file)
        file = './dumpData/NCHW/'
        mkdir(file)

    expose_model = expose_node_outputs(args.model_file, False)
    #expose_model = args.input_model
    print("Generate Output Data ....")
    runOnnxRuntime(expose_model, images, dump_bin, preprocess_funcs)
    print("Onnx dump data done. See results in ./dumpData")


if __name__ == '__main__':
    NHWC = True #onnx is NCHW
    main()
