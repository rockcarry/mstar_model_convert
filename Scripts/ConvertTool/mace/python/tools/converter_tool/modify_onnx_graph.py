import onnx
import pdb
import pickle
import struct
import numpy as np
import argparse
import onnxruntime
import copy

def get_float_data(tensor,tensor_dict):
    if tensor.float_data != []:
        tensor_dict[tensor.name] = tensor.float_data
    elif tensor.raw_data != []:
        length = 1
        for i in range(len(tensor.dims)):
            length = length*tensor.dims[i]
        data_raw = struct.unpack('f'*length,tensor.raw_data)
        tensor_dict[tensor.name] = list(data_raw)[0]
    return tensor_dict

def get_uint8_data(tensor,tensor_dict):
    if tensor.int32_data != []: ###########
        tensor_dict[tensor.name] = tensor.int32_data
    elif tensor.raw_data != []:
        length = 1
        for i in range(len(tensor.dims)):
            length = length*tensor.dims[i]
        data_raw = struct.unpack('b'*length,tensor.raw_data)
        tensor_dict[tensor.name] = list(data_raw)[0]
    return tensor_dict

def get_int8_data(tensor,tensor_dict):
    if tensor.int32_data != []: ###########
        tensor_dict[tensor.name] = tensor.int32_data
    elif tensor.raw_data != []:
        length = 1
        for i in range(len(tensor.dims)):
            length = length*tensor.dims[i]
        data_raw = struct.unpack('b'*length,tensor.raw_data)
        tensor_dict[tensor.name] = list(data_raw)[0]
    return tensor_dict

def get_uint16_data(tensor,tensor_dict):
    if tensor.int32_data != []: ###########
        tensor_dict[tensor.name] = tensor.int32_data
    elif tensor.raw_data != []:
        length = 1
        for i in range(len(tensor.dims)):
            length = length*tensor.dims[i]
        data_raw = struct.unpack('h'*length,tensor.raw_data)
        tensor_dict[tensor.name] = list(data_raw)[0]
    return tensor_dict

def get_int16_data(tensor,tensor_dict):
    if tensor.int32_data != []: ###########
        tensor_dict[tensor.name] = tensor.int32_data
    elif tensor.raw_data != []:
        length = 1
        for i in range(len(tensor.dims)):
            length = length*tensor.dims[i]
        data_raw = struct.unpack('h'*length,tensor.raw_data)
        tensor_dict[tensor.name] = list(data_raw)[0]
    return tensor_dict

def createGraphMemberMap_node(graph_member_list):
    member_map=dict()
    for idx,n in enumerate(graph_member_list):
        if n.op_type not in member_map.keys():
            member_map[n.op_type] = 1
        else:
            member_map[n.op_type] = member_map[n.op_type] + 1
    return member_map

def createGraphMemberMap_initializer(graph_member_list):
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name] = n
    return member_map


def CheckModelStructure(orimodel):
    onnx_model = onnx.load(orimodel)
    graph_info = onnx_model.graph
    nodes_info = graph_info.node
    op_statistics = createGraphMemberMap_node(nodes_info)
    if 'QuantizeLinear' in op_statistics.keys()  or 'DequantizeLinear' in op_statistics.keys():
        return True
    else:
        return False

class ModifyGraph(object):
    def __init__(self, net, pkl_replica_map):
        self._net = net
        self._pkl_replica_map = pkl_replica_map


    def save_quantitative_info(self, graph,Q_input0_name,Q_scale_name,Q_zero_name):
        # NOTE:Onnx proto data_type as follows
        #    | FLOAT = 1;   // float   |
        #    | UINT8 = 2;   // uint8_t |
        #    | INT8 = 3;    // int8_t  |
        #    | UINT16 = 4;  // uint16_t|
        #    | INT16 = 5;   // int16_t |
        #    | INT32 = 6;   // int32_t |
        #    | INT64 = 7;   // int64_t |
        #    | STRING = 8;  // string  |
        #    | BOOL = 9;    // bool    |

        # 1、Create dictionary to store quantitative information
        # --- scale dict and zero dict have a one-to-one correspondence
        # --- quant dict record quant level of each pair of quantitative information
        onnx_scale_const_tensors ,onnx_zero_const_tensors, quant_level= dict(),dict(),dict()

        # 2、Initialize for zero dictionary
        # --- zero_point tensor may be empty, when it is empty,
        # --- the default quantization level is uint8 and zero_point is 0,
        # --- so it is necessary to first create a dictionary corresponding to the non-existent zero_point tensor
        for i in range(len(Q_zero_name)):
            if Q_zero_name[i].lower().find( 'sgs_default') == 0:
                onnx_zero_const_tensors[Q_zero_name[i]] = 0
                quant_level[Q_zero_name[i]] = 'uint8'
            else:
                onnx_zero_const_tensors[Q_zero_name[i]] = 0

        # 3、Read quantitative information and store them in their respective dict
        for _, cc in enumerate(graph.initializer):
            # get scale info
            if cc.name in Q_scale_name:
                # float32
                if cc.data_type == 1:
                    onnx_scale_const_tensors = get_float_data(cc,onnx_scale_const_tensors)
                # int8
                elif cc.data_type == 3:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_scale_const_tensors = get_int8_data(cc,onnx_scale_const_tensors)
                # uint8
                elif cc.data_type == 2:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_scale_const_tensors = get_uint8_data(cc,onnx_scale_const_tensors)
                # int16
                elif cc.data_type == 5:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_scale_const_tensors = get_int16_data(cc,onnx_scale_const_tensors)
                # uint16
                elif cc.data_type == 4:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_scale_const_tensors = get_uint16_data(cc,onnx_scale_const_tensors)

            # get zero info and get the quantization level according to the data type, assign to quant dict
            elif cc.name in Q_zero_name:
                # float32
                if cc.data_type == 1:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_zero_const_tensors = get_float_data(cc,onnx_zero_const_tensors)
                # int8
                elif cc.data_type == 3:
                    onnx_zero_const_tensors = get_int8_data(cc,onnx_zero_const_tensors)
                    quant_level[cc.name] = 'int8'
                # uint8
                elif cc.data_type == 2:
                    onnx_zero_const_tensors = get_uint8_data(cc,onnx_zero_const_tensors)
                    quant_level[cc.name] = 'uint8'
                # int16
                elif cc.data_type == 5:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_zero_const_tensors = get_int16_data(cc,onnx_zero_const_tensors)
                    quant_level[cc.name] = 'int16'
                # uint16
                elif cc.data_type == 4:
                    mace_check(False, "Not supported tensor type: %s" % cc.data_type)
                    onnx_zero_const_tensors = get_uint16_data(cc,onnx_zero_const_tensors)
                    quant_level[cc.name] = 'uint16'

        # 4、Calculate the min max value of the original data according to the quantization information and write it into pkl
        pkl_info = []
        name_set = set()
        name_duplication_times = dict()
        match_scale_zero_name = list(zip(Q_scale_name,Q_zero_name))
        for i in range(len(match_scale_zero_name)):
            pkl_info_dict = dict()
            current_match = match_scale_zero_name[i]
            current_scale_value = onnx_scale_const_tensors[current_match[0]]
            current_zero_value = onnx_zero_const_tensors[current_match[1]]
            if quant_level[current_match[1]] == 'int8':
                ori_max = (127-current_zero_value)*current_scale_value
                ori_min = ori_max - current_scale_value*(127-(-128))
                if Q_input0_name[i] not in name_set:
                    name_set.add(Q_input0_name[i])
                    pkl_info_dict['name'] = Q_input0_name[i]
                    name_duplication_times[Q_input0_name[i]] = -1
                else:
                    name_duplication_times[Q_input0_name[i]] += 1
                    if Q_input0_name[i] in self._pkl_replica_map.keys():
                        pkl_info_dict['name'] = self._pkl_replica_map[Q_input0_name[i]][name_duplication_times[Q_input0_name[i]]]
                    else:
                        pkl_info_dict['name'] = Q_input0_name[i]
                pkl_info_dict['min'] = [ori_min]
                pkl_info_dict['max'] = [ori_max]
                pkl_info_dict['bit'] = 8
                pkl_info.append(pkl_info_dict)

            if quant_level[current_match[1]] == 'uint8':
                ori_max = (255-current_zero_value)*current_scale_value
                ori_min = ori_max - current_scale_value*(255-(0))
                if Q_input0_name[i] not in name_set:
                    name_set.add(Q_input0_name[i])
                    pkl_info_dict['name'] = Q_input0_name[i]
                    name_duplication_times[Q_input0_name[i]] = -1
                else:
                    name_duplication_times[Q_input0_name[i]] += 1
                    if Q_input0_name[i] in self._pkl_replica_map.keys():
                        pkl_info_dict['name'] = self._pkl_replica_map[Q_input0_name[i]][name_duplication_times[Q_input0_name[i]]]
                    else:
                        pkl_info_dict['name'] = Q_input0_name[i]
                pkl_info_dict['min'] = [ori_min]
                pkl_info_dict['max'] = [ori_max]
                pkl_info_dict['bit'] = 8
                pkl_info.append(pkl_info_dict)

            if quant_level[current_match[1]] == 'int16':
                ori_max = (32767-current_zero_value)*current_scale_value
                ori_min = ori_max - current_scale_value*(32767-(-32768))
                if Q_input0_name[i] not in name_set:
                    name_set.add(Q_input0_name[i])
                    pkl_info_dict['name'] = Q_input0_name[i]
                    name_duplication_times[Q_input0_name[i]] = -1
                else:
                    name_duplication_times[Q_input0_name[i]] += 1
                    if Q_input0_name[i] in self._pkl_replica_map.keys():
                        pkl_info_dict['name'] = self._pkl_replica_map[Q_input0_name[i]][name_duplication_times[Q_input0_name[i]]]
                    else:
                        pkl_info_dict['name'] = Q_input0_name[i]
                pkl_info_dict['min'] = [ori_min]
                pkl_info_dict['max'] = [ori_max]
                pkl_info_dict['bit'] = 16
                pkl_info.append(pkl_info_dict)

            if quant_level[current_match[1]] == 'uint16':
                ori_max = (65535-current_zero_value)*current_scale_value
                ori_min = ori_max - current_scale_value*(65535-(0))
                if Q_input0_name[i] not in name_set:
                    name_set.add(Q_input0_name[i])
                    pkl_info_dict['name'] = Q_input0_name[i]
                    name_duplication_times[Q_input0_name[i]] = -1
                else:
                    name_duplication_times[Q_input0_name[i]] += 1
                    if Q_input0_name[i] in self._pkl_replica_map.keys():
                        pkl_info_dict['name'] = self._pkl_replica_map[Q_input0_name[i]][name_duplication_times[Q_input0_name[i]]]
                    else:
                        pkl_info_dict['name'] = Q_input0_name[i]
                pkl_info_dict['min'] = [ori_min]
                pkl_info_dict['max'] = [ori_max]
                pkl_info_dict['bit'] = 16
                pkl_info.append(pkl_info_dict)

        model_lower = self._net.split('.onnx')[0]
        file_name = model_lower + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(pkl_info, f)

        with open(file_name, 'rb') as f:
            a = pickle.load(f)
            print(a)

    def get_op_mapping_info(self, nodes,graph):
        replace_input0_dict,replace_input1_dict,replace_input2_dict = {},{},{}
        remove_q_list,remove_dq_list = [],[]
        Q_input0_name,Q_scale_tensor_name,Q_zero_tensor_name = [],[],[]
        str_num = -1 # Used to record that the zero_point tensor is empty

        # 1、Find the quantization operator corresponding to operator A,
        # --- only supports conv2d and maxpool
        for i in range(len(nodes)):
            current_node = nodes[i]
            # (1) conv2d
            # --- find QDQ for ib
            # --- find QDQ for kb
            # --- find QDQ for bias if bias exitst
            if current_node.op_type == "Conv":
                existBias = False
                conv_ib = current_node.input[0]
                conv_kb = current_node.input[1]
                if len(current_node.input) == 3:
                    conv_bias = current_node.input[2]
                    existBias = True
                for j in range(len(nodes)):
                    DQ_node = nodes[j]
                    # find QDQ for ib
                    if DQ_node.op_type == "DequantizeLinear" and DQ_node.output[0] == conv_ib:
                        Q_node = nodes[j-1]
                        if Q_node.op_type == "QuantizeLinear" and Q_node.output[0] == DQ_node.input[0]:
                            replace_input0_dict[i] = Q_node.input[0]
                            remove_q_list.append(j-1)
                            remove_dq_list.append(j)
                            Q_scale_tensor_name.append(Q_node.input[1])
                            if len(Q_node.input) > 2:
                                Q_zero_tensor_name.append(Q_node.input[2])
                            else:
                                # In order to keep the length of the scale and zero lists consistent to find a matching relationship,
                                # if the zero point tensor is empty,
                                # it will be recorded as a new name and will be assigned a value of 0 in the future
                                str_num += 1
                                Q_zero_tensor_name.append('sgs_default'+str(str_num))
                            Q_input0_name.append(Q_node.input[0])
                        else:
                            print("'QuantizeLinear' and 'DequantizeLinear' must be used adjacently one-to-one!")
                            return False
                    # find QDQ for kb
                    if DQ_node.op_type == "DequantizeLinear" and DQ_node.output[0] == conv_kb:
                        Q_node = nodes[j-1]
                        if Q_node.op_type == "QuantizeLinear" and Q_node.output[0] == DQ_node.input[0]:
                            replace_input1_dict[i] = Q_node.input[0]
                            remove_q_list.append(j-1)
                            remove_dq_list.append(j)
                            Q_scale_tensor_name.append(Q_node.input[1])
                            if len(Q_node.input) > 2:
                                Q_zero_tensor_name.append(Q_node.input[2])
                            else:
                                str_num += 1
                                Q_zero_tensor_name.append('sgs_default'+str(str_num))
                            Q_input0_name.append(Q_node.input[0])
                        else:
                            print("'QuantizeLinear' and 'DequantizeLinear' must be used adjacently one-to-one!")
                            return False
                    # find QDQ for bias
                    if existBias:
                        if DQ_node.op_type == "DequantizeLinear" and DQ_node.output[0] == conv_bias:
                            Q_node = nodes[j-1]
                            if Q_node.op_type == "QuantizeLinear" and Q_node.output[0] == DQ_node.input[0]:
                                replace_input2_dict[i] = Q_node.input[0]
                                remove_q_list.append(j-1)
                                remove_dq_list.append(j)
                                Q_scale_tensor_name.append(Q_node.input[1])
                                if len(Q_node.input) > 2:
                                    Q_zero_tensor_name.append(Q_node.input[2])
                                else:
                                    str_num += 1
                                    Q_zero_tensor_name.append('sgs_default'+str(str_num))
                                Q_input0_name.append(Q_node.input[0])
                            else:
                                print("'QuantizeLinear' and 'DequantizeLinear' must be used adjacently one-to-one!")
                                return False
            # (2) maxpool
            # --- find QDQ for input
            if current_node.op_type == "MaxPool":
                maxpool_input = current_node.input[0]
                for j in range(len(nodes)):
                    DQ_node = nodes[j]
                    if DQ_node.op_type == "DequantizeLinear" and DQ_node.output[0] == maxpool_input:
                        Q_node = nodes[j-1]
                        if Q_node.op_type == "QuantizeLinear" and Q_node.output[0] == DQ_node.input[0]:
                            replace_input0_dict[i] = Q_node.input[0]
                            remove_q_list.append(j-1)
                            remove_dq_list.append(j)
                            Q_scale_tensor_name.append(Q_node.input[1])
                            if len(Q_node.input) > 2:
                                Q_zero_tensor_name.append(Q_node.input[2])
                            else:
                                str_num += 1
                                Q_zero_tensor_name.append('sgs_default'+str(str_num))
                            Q_input0_name.append(Q_node.input[0])
                        else:
                            print("'QuantizeLinear' and 'DequantizeLinear' must be used adjacently one-to-one!")
                            return False
        # 2、generate pkl file
        self.save_quantitative_info(graph,Q_input0_name,Q_scale_tensor_name,Q_zero_tensor_name)

        return replace_input0_dict,replace_input1_dict,remove_q_list,remove_dq_list


    def run(self):
        model_path = self._net
        onnx_with_QDQ_model = onnx.load(model_path)
        graph_info = onnx_with_QDQ_model.graph
        nodes_info = graph_info.node
        ori_op_statistics = createGraphMemberMap_node(nodes_info)
        # 1、get Qop list and DQop list
        replace_input0_dict,replace_input1_dict,remove_q_list,remove_dq_list = self.get_op_mapping_info(nodes_info,graph_info)

        # 2、do remove and replace input with reverse order,
        # otherwise the following idx will out of range if the idx following the positive order
        initializer_map = createGraphMemberMap_initializer(graph_info.initializer)
        deleted_const_list = []
        for i in range(len(nodes_info)-1,-1,-1):
            # replace input tensor name
            for key,value in replace_input0_dict.items():
                if i == key:
                    nodes_info[i].input[0] = value
            for key,value in replace_input1_dict.items():
                if i == key:
                    nodes_info[i].input[1] = value
            # remove discard const tensor
            if i in remove_dq_list:
                if len(nodes_info[i].input) > 2:
                    if nodes_info[i].input[2] not in deleted_const_list:
                        graph_info.initializer.remove(initializer_map[nodes_info[i].input[2]])
                        deleted_const_list.append(nodes_info[i].input[2])
                if nodes_info[i].input[1] not in deleted_const_list:
                    graph_info.initializer.remove(initializer_map[nodes_info[i].input[1]])
                    deleted_const_list.append(nodes_info[i].input[1])
            if i in remove_q_list:
                if len(nodes_info[i].input) > 2:
                    if nodes_info[i].input[2] not in deleted_const_list:
                        graph_info.initializer.remove(initializer_map[nodes_info[i].input[2]])
                        deleted_const_list.append(nodes_info[i].input[2])
                if nodes_info[i].input[1] not in deleted_const_list:
                    graph_info.initializer.remove(initializer_map[nodes_info[i].input[1]])
                    deleted_const_list.append(nodes_info[i].input[1])
            # remove discard op
            if i in remove_dq_list:
                graph_info.node.remove(nodes_info[i])
            if i in remove_q_list:
                graph_info.node.remove(nodes_info[i])

        modify_op_statistics = createGraphMemberMap_node(nodes_info)
        print("------------------------------------------------------")
        print("The original model operator statistics are as follows:")
        print(ori_op_statistics)
        print("------------------------------------------------------")
        print("The newl model operator statistics are as follows:")
        print(modify_op_statistics)
        print("------------------------------------------------------")
        # 3、save model
        onnx.checker.check_model(onnx_with_QDQ_model)
        onnx.save(onnx_with_QDQ_model, 'modify.onnx')
        return 'modify.onnx'
