# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import numpy as np
import argparse
import pickle
from calibrator_custom import utils
from calibrator_custom import printf
from calibrator_custom import rpc_simulator


def batch_image_generator(image_list, preprocess_funcs):
    imgs = []
    for idx, preprocess_func in enumerate(preprocess_funcs):
        ii = []
        for img in image_list[idx]:
            ii.append(preprocess_func(img, False))
        imgs.append(ii)
    return imgs


class Net(calibrator_custom.RPC_Simulator):
    def __init__(self, model_path, batch_size):
        super().__init__()
        self.model = rpc_simulator.simulator(model_path, batch=batch_size)

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            if in_details[idx]['batch_mode'] == 'n_buf':
                for i in range(in_details[idx]['batch']):
                    self.model.set_input(idx, utils.convert_to_input_formats(x[idx][i], in_details[idx]), i)
            else:
                data = np.concatenate(x[idx])
                self.model.set_input(idx, utils.convert_to_input_formats(data, in_details[idx]))
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            if out_details[idx]['batch_mode'] == 'n_buf':
                batch_res = []
                for i in range(out_details[idx]['batch']):
                    result = self.model.get_output(idx, i)
                    batch_res.append(result)
            else:
                batch_res = self.model.get_output(idx)
            result_list.append(batch_res)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='Multi Batch RPC Simulator Tool')
    parser.add_argument('--host', type=str, required=True,
                        help='IPU Server host.')
    parser.add_argument('--port', type=int, required=True,
                        help='IPU Server port.')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image path (colon-separated for multi-batch, comma-separated for multi-inputs)')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Offline Model path.')
    parser.add_argument('-b', '--batch', type=int, required=True,
                        help='Set amount of batch size.')
    parser.add_argument('-n', '--preprocess', type=str, required=True,
                        help='Name of model to select image preprocess method (comma-separated for multi-inputs)')

    return parser.parse_args()


def classify(output, top_k=5):
    labels = {i: str(i) for i in range(1001)}
    output_0 = np.squeeze(output)
    ordered = np.argpartition(-output_0, top_k)
    results = [(i, output_0[i]) for i in ordered[:top_k]]
    results.sort(key=lambda x: -x[1])
    for i in range(top_k):
        label_id, prob = results[i]
        printf('%s %.6f' % (labels[label_id], prob))
    printf('\n')


def write_txt(f, data):
    for num, value in enumerate(list(data.flat)):
        f.write('{:.6f}  '.format(value))
        if (num + 1) % 16 == 0:
            f.write('\n')


def write_output(model_path, result_list, out_details):
    txt_name = 'multi_batch_{}batch_{}_{}.txt'.format(
        out_details[0]['batch'], out_details[0]['batch_mode'], os.path.basename(model_path))
    with open(txt_name, 'w') as f:
        for i, result in enumerate(result_list):
            f.write('{} Tensor:\n{{\n'.format(out_details[i]['name']))
            f.write('tensor dim:{},  shape:[{}],  batch_size:{},  batch_mode:{}\n'
                .format(out_details[i]['shape'].shape[0], ' '.join(str(v) for v in list(out_details[i]['shape'].flat)),
                out_details[i]['batch'], out_details[i]['batch_mode']))
            if out_details[i]['batch_mode'] == 'n_buf':
                for batch_idx, batch_result in enumerate(result):
                    f.write('batch index {} tensor data:\n'.format(batch_idx))
                    write_txt(f, batch_result)
                    f.write('\n')
            else:
                f.write('tensor data:\n')
                write_txt(f, result)
            f.write('\n}\n\n')
    printf('Output saved in {}'.format(txt_name))


def main():
    if utils.get_sdk_version() in ['1', 'Q_0']:
        raise ValueError('This chip Do Not Support multi-batch!')
    args = arg_parse()
    host = args.host
    port = args.port
    image_path = args.image
    model_path = args.model
    model_name = args.preprocess
    batch = args.batch

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    preprocess_funcs = [utils.image_preprocess_func(n) for n in model_name.split(',')]
    img_list = [[os.path.abspath(img) for img in imgs.split(':')] for imgs in image_path.split(',')]
    for imgs in img_list:
        if len(imgs) != batch:
            raise ValueError('Got different num of batch size and images!')

    if len(preprocess_funcs) != len(img_list):
        raise ValueError('Got different num of preprocess_methods and images!')

    img_gen = batch_image_generator(img_list, preprocess_funcs)
    rpc_simulator.connect(host, port)
    net = Net(model_path, batch)
    printf(str(net))
    result = net(img_gen)

    write_output(model_path, result, net.model.get_output_details())


if __name__ == '__main__':
    main()
