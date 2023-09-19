import calibrator_custom
import sys
import argparse
import os
import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='Show tensor info Tool')
    subparsers = parser.add_subparsers(help='Parameter usage instructions')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Cmodel model path.')
    parser.add_argument('--input_config', type=str, required=True,
                        help='Input config path.')
    parser.add_argument('--quant_level', type=str, default='ALL',
                        choices=['16', '8', '4','ALL'],
                        help='Indicate Quantilization level.')
    return parser.parse_args()

def get_details(model_path, input_config):
    c = calibrator_custom.calibrator(model_path, input_config)
    ops = c.get_op_details(['CONV_2D','DEPTHWISE_CONV_2D','CONV_3D','BATCH_MATMUL','GroupConv'])
    tensors = c.get_tensor_details()
    ops_input_list = []
    ops_weight_list = []
    op_idx = 0
    for op in ops:
        op_input = op['input_tensors'][0]
        op_weight = op['input_tensors'][1]
        ops_input_list.append(op_input)
        ops_weight_list.append(op_weight)
        op_idx += 1
    if len(ops_input_list) != len(ops_weight_list):
        print('input num must be same as weight num!')
        sys.exit(0)

    tensors_dict = dict()
    for tensor in tensors:
        tensors_dict[tensor['name']] = tensor

    return ops_input_list,ops_weight_list,tensors_dict

def save_results(quant_level,ops_qtype_INT16_list,ops_qtype_UINT8_list,ops_qtype_UINT4_list):
    parnet = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    save_path = os.path.join(parnet, 'Scripts','Miscellaneous')
    isExists=os.path.exists(save_path)
    if isExists:
        save_txt = os.path.join(save_path, 'Tensor_qtype'+ '.txt')
    else:
        os.makedirs(save_path)
        save_txt = os.path.join(save_path, 'Tensor_qtype'+ '.txt')

    with open(save_txt, 'w') as f:
        if quant_level == 'ALL':
            head1 = "//[qtype_INT16]:" + "tensorNum:" + str(2*len(ops_qtype_INT16_list)) + ",opNum:" + str(len(ops_qtype_INT16_list)) +  "\n"
            f.write(head1)
            for i in range(len(ops_qtype_INT16_list)):
                item = 0
                for key,value in ops_qtype_INT16_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + ',' )
                        item +=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1

            head2 = "//[qtype_UINT8]:" + "tensorNum:" + str(2*len(ops_qtype_UINT8_list)) + ",opNum:" + str(len(ops_qtype_UINT8_list)) +  "\n"
            f.write(head2)
            for i in range(len(ops_qtype_UINT8_list)):
                item = 0
                for key,value in ops_qtype_UINT8_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "," )
                        item+=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1

            head3 = "//[qtype_UINT4]:" + "tensorNum:" + str(2*len(ops_qtype_UINT4_list)) + ",opNum:" + str(len(ops_qtype_UINT4_list)) +  "\n"
            f.write(head3)
            for i in range(len(ops_qtype_UINT4_list)):
                item = 0
                for key,value in ops_qtype_UINT4_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "," )
                        item+=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1

        elif quant_level == '16':
            head1 = "//[qtype_INT16]:" + "tensorNum:" + str(2*len(ops_qtype_INT16_list)) + ",opNum:" + str(len(ops_qtype_INT16_list)) +  "\n"
            f.write(head1)
            for i in range(len(ops_qtype_INT16_list)):
                item = 0
                for key,value in ops_qtype_INT16_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + ',' )
                        item +=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1
        elif quant_level == '8':
            head2 = "//[qtype_UINT8]:" + "tensorNum:" + str(2*len(ops_qtype_UINT8_list)) + ",opNum:" + str(len(ops_qtype_UINT8_list)) +  "\n"
            f.write(head2)
            for i in range(len(ops_qtype_UINT8_list)):
                item = 0
                for key,value in ops_qtype_UINT8_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "," )
                        item+=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1
        elif quant_level == '4':
            head3 = "//[qtype_UINT4]:" + "tensorNum:" + str(2*len(ops_qtype_UINT4_list)) + ",opNum:" + str(len(ops_qtype_UINT4_list)) +  "\n"
            f.write(head3)
            for i in range(len(ops_qtype_UINT4_list)):
                item = 0
                for key,value in ops_qtype_UINT4_list[i].items():
                    if item%2==0:
                        data = f.write('IbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "," )
                        item+=1
                    else:
                        data = f.write('KbName:' + '[' + str(key) + ']' + ',Qtype:' + '[' + str(value)+ ']' + "\n" )
                        item+=1

    print("=== create TXT done, file [Tensor_qtype.txt] at:  %s " % (save_txt))

if __name__ == '__main__':
    args = arg_parse()
    model_path = args.model
    input_config = args.input_config
    quant_level = args.quant_level
    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    ops_input_list,ops_weight_list,tensors_dict = get_details(model_path, input_config)
    ops_qtype_INT16_list = []
    ops_qtype_UINT8_list = []
    ops_qtype_UINT4_list = []
    for ib_name,kb_name in zip(ops_input_list,ops_weight_list):
        ops_qtype_INT16_dict = dict()
        ops_qtype_UINT8_dict = dict()
        ops_qtype_UINT4_dict = dict()
        ib_qtype = tensors_dict[ib_name]['qtype']
        kb_qtype = tensors_dict[kb_name]['qtype']
        # INT16
        if ib_qtype == 'INT16' and kb_qtype == 'INT16':
            ops_qtype_INT16_dict[ib_name] = ib_qtype
            ops_qtype_INT16_dict[kb_name] = kb_qtype
            ops_qtype_INT16_list.append(ops_qtype_INT16_dict)
        # UINT8
        if ib_qtype == 'UINT8' and kb_qtype == 'INT8':
            ops_qtype_UINT8_dict[ib_name] = ib_qtype
            ops_qtype_UINT8_dict[kb_name] = kb_qtype
            ops_qtype_UINT8_list.append(ops_qtype_UINT8_dict)
        # UINT4
        if ib_qtype == 'UINT8' and kb_qtype == 'INT4':
            ops_qtype_UINT4_dict[ib_name] = ib_qtype
            ops_qtype_UINT4_dict[kb_name] = kb_qtype
            ops_qtype_UINT4_list.append(ops_qtype_UINT4_dict)
        if ib_qtype == 'UINT4' and kb_qtype == 'INT4':
            ops_qtype_UINT4_dict[ib_name] = ib_qtype
            ops_qtype_UINT4_dict[kb_name] = kb_qtype
            ops_qtype_UINT4_list.append(ops_qtype_UINT4_dict)
        if ib_qtype == 'UINT4' and kb_qtype == 'INT8':
            ops_qtype_UINT4_dict[ib_name] = ib_qtype
            ops_qtype_UINT4_dict[kb_name] = kb_qtype
            ops_qtype_UINT4_list.append(ops_qtype_UINT4_dict)

    save_results(quant_level,ops_qtype_INT16_list,ops_qtype_UINT8_list,ops_qtype_UINT4_list)



