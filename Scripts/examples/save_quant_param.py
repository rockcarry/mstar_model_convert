import calibrator_custom
import argparse
import json
import pickle
import os

def arg_parse():
    parser = argparse.ArgumentParser(description='Save import quant param Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Fixed Model path.')
    parser.add_argument('--output_mode', type=str, default='JSON',
                        choices=['JSON', 'Pickle'],
                        help='Indicate Quant param file format.')

    return parser.parse_args()


def convert_dtype(dtype):
    if dtype in ['UINT8', 'INT8']:
        return 8
    elif dtype in ['INT16', 'INT32', 'FLOAT32']:
        return 16
    elif dtype in ['UINT4', 'INT4']:
        return 4
    else:
        raise ValueError('Unknown dtype: {}'.format(dtype))


def main():
    args = arg_parse()
    model_path = args.model
    output_mode = args.output_mode

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such model: {}'.format(model_path))
    model = calibrator_custom.fixed_simulator(model_path)
    quant_data = model.get_tensor_details()
    for idx in range(len(quant_data) - 1, -1, -1):
        if len(quant_data[idx]['max']) == 0 or len(quant_data[idx]['min']) == 0:
            del quant_data[idx]
            continue
        del quant_data[idx]['quantization']
        del quant_data[idx]['shape']
        quant_data[idx]['bit'] = convert_dtype(quant_data[idx]['dtype'])
        del quant_data[idx]['dtype']

    if output_mode == 'JSON':
        quant_file_name = model_path + '.json'
        with open(quant_file_name, 'w') as f:
            json.dump(quant_data, f, indent=4)
    else:
        quant_file_name = model_path + '.pkl'
        with open(quant_file_name, 'wb') as f:
            pickle.dump(quant_data, f)
    print('Quant param saved in {}'.format(quant_file_name))

if __name__ == '__main__':
    main()
