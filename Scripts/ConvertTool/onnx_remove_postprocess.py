import onnx
import pdb
import argparse
import copy
from onnx import helper, checker, TensorProto
from mace.python.tools.onnx import symbolic_shape_infer

def remove_node_and_save(input_model_path, output_model_path, allshape, remove_node_name):
    onnx_model = onnx.load(input_model_path)
    graph = onnx_model.graph
    node  = graph.node
    node_copy = copy.deepcopy(node)
    lenNode = len(remove_node_name)
    #pdb.set_trace()
    begin = 0

    dictNode_name_index = {}
    node_copy = copy.deepcopy(node)
    for i in range(len(node_copy)):
        if node_copy[i].name in remove_node_name:
            #pdb.set_trace()
            begin = i
            dictNode_name_index[node_copy[i].name] = i

    for i in range(len(node_copy)):
        if i <= begin:
            continue
        graph.node.remove(node_copy[i])

    for i in range(len(graph.output)):#remove all of src outputs
        graph.output.pop()


    output_shape = {}
    for i in range(lenNode):
        node_number = dictNode_name_index[remove_node_name[i]]
        name = graph.node[node_number].output[0]#get output node name
        shape = allshape[name]#get shape from a dict
        new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)#set a output node
        graph.output.extend([new_nv])
    onnx.checker.check_model(onnx_model)
    #remove unuseful Input
    while(len(onnx_model.graph.input) > 1):
        onnx_model.graph.input.remove(onnx_model.graph.input[1])
    #remove unuseful tensor
    node_copy = copy.deepcopy(onnx_model.graph.node)
    node_of_tensor_name = []

    for index, node in enumerate(node_copy):
        for i in range(len(node.input)):
            node_of_tensor_name.append(node.input[i])

    initializer_copy = copy.deepcopy(onnx_model.graph.initializer)
    for index, tensor in enumerate(initializer_copy):
        #print(tensor.name)
        if not tensor.name in node_of_tensor_name:
            onnx_model.graph.initializer.remove(tensor)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_model_path)

def getAllShape(model):

    graph_shapes_dict = {}
    onnx_model = onnx.load(model)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input ONNX model final_xxx.onnx')
    parser.add_argument('output_model', help='Output ONNX model')
    parser.add_argument('--node', help='Given node name, for example "Concat_470')
    args = parser.parse_args()
    #cal the shape of input_model
    print("Starting ShapeInference")
    final_onnx_model = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(args.input_model,"./final.onnx")

    node_name = []
    if args.node is not None:
        if ',' not in args.node:#only one of node
            node_name.append(str(args.node))
        else:
            node_name = args.node.split(",")
    layer_names_shapes = getAllShape("./final.onnx")
    print("Removing Postprocess...")
    remove_node_and_save("final.onnx", args.output_model, layer_names_shapes, node_name)
    print("Remove Done ! Please check "+ args.output_model)
if __name__ == '__main__':
    main()