import calibrator_custom
import argparse
import os
import numpy as np

class Offline_Model(calibrator_custom.SIM_Simulator):
    def __init__(self, model_path, batch_size=1):
        super().__init__()
        if calibrator_custom.__version__[0] in ['S']:
            self.offline_model = calibrator_custom.offline_simulator(model_path, batch=batch)
        else:
            self.offline_model = calibrator_custom.offline_simulator(model_path)

    def forward(self, x):
        out_details = self.model.get_output_details()
        self.offline_model.set_input(0, x)
        self.offline_model.invoke()
        result_list = []
        for idx in range(len(out_details)):
            result = self.offline_model.get_output(idx)
            # for Fixed and Offline model
            if result.shape[-1] != out_details[idx]['shape'][-1]:
                result = result[..., :out_details[idx]['shape'][-1]]
            if out_details[idx]['dtype'] == np.int16:
                scale, _ = out_details[idx]['quantization']
                result = np.dot(result, scale)
            result_list.append(result)
        return result_list


def arg_parse():
    parser = argparse.ArgumentParser(description='Show img info Tool')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Offline Model path.')
    parser.add_argument('--version', action='store_true', required=False,
                        help='Print SDK version info.')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    model_path = args.model

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))


    if 'SGS_IPU_DIR' in os.environ:
        Project_path = os.environ['SGS_IPU_DIR']
        parse_img = os.path.join(Project_path, 'bin/parse_img')
    elif 'OUT_DIR' in os.environ:
        Project_path = os.environ['OUT_DIR']
        parse_img = os.path.join(Project_path, 'bin/parse_img')
    else:
        raise OSError('Run source cfg_env.sh in SGS_IPU_SDK directory.')
    os.system('{} {} FALSE > offline_img.log'.format(parse_img, model_path))
    with open('offline_img.log', 'r') as f:
        offline_info = f.readlines()

    buf_size = 0
    max_batch_num = 1
    batches_tmp = []
    batches = []
    for line in offline_info:
        if 'SGS_SDK_VERSION' in line:
            print('Offline IMG Version: {}'.format(line.split(' ')[-1].strip()))
        if 'SGS_MODEL_SIZE' in line:
            print('Offline Model Size: {}'.format(line.split(' ')[-1].strip()))
        if 'BATCH' in line[:5]:
            batches_tmp.append(int(line.split(':')[-1].strip().split(',')[0]))
        if 'SGS_BUFFER_SIZE' in line:
            buf_size += int(line.split(' ')[-1])
        if 'USER_MAX_BATCH_NUM' in line:
            print('Offline Model Max Batch Numer: {}'.format(line.split(' ')[-1].strip()))
    if args.version:
        print('SDK_Version_Info: {}'.format(calibrator_custom.__version__))

    # delete duplicate batches
    for i in batches_tmp:
        if i not in batches:
            batches.append(i)

    batch = batches[0] if len(batches) > 0 else 1
    offline_model = Offline_Model(model_path, batch)

    if len(batches) > 0:
        if offline_model.offline_model.get_input_details()[0]['batch_mode'] == 'n_buf':
            print('Offline Model Suggested Batch: [{}]'.format(', '.join([str(i) for i in batches])))
        else:
            print('Offline Model Supported Batch: [{}]'.format(', '.join([str(i) for i in batches])))

    print('Offline Model Variable Buffer Size: {}\n'.format(buf_size))
    os.remove('offline_img.log')
    print(offline_model)
