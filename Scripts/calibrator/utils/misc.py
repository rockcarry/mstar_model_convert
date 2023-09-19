# -*- coding: utf-8 -*-

import time
import sys
import os
import cv2
import shutil
import importlib
import numpy as np
import calibrator_custom
from calibrator_custom import utils
from calibrator_custom import printf
from multiprocessing import Pool
import gc

cv2.setNumThreads(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
SHOW_LOG = False


class ShowProcess(object):
    def __init__(self, max_steps, max_arrow=50):
        self.max_steps = max_steps
        self.i = 0
        self.max_arrow = max_arrow
        self.start = time.time()
        self.eta = 0.0
        self.total_time = 0.0
        self.last_time = self.start

    def elapsed_time(self):
        self.last_time = time.time()
        return self.last_time - self.start

    def calc_eta(self):
        elapsed = self.elapsed_time()
        if self.i == 0 or elapsed < 0.001:
            return None
        rate = float(self.i) / elapsed
        self.eta = (float(self.max_steps) - float(self.i)) / rate

    def get_time(self, _time):
        if (_time < 86400):
            return time.strftime("%H:%M:%S", time.gmtime(_time))
        else:
            s = (str(int(_time // 3600)) + ':' +
                 time.strftime("%M:%S", time.gmtime(_time)))
            return s

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        self.calc_eta()
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        if num_arrow < 2:
            process_bar = '\r' + '[' + '>' * num_arrow + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        elif num_arrow < self.max_arrow:
            process_bar = '\r' + '[' + '=' * (num_arrow-1) + '>' + ' ' * num_line + ']' \
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        else:
            process_bar = '\r' + '[' + '=' * num_arrow + ' ' * num_line + ']'\
                          + '%.2f' % percent + '%' + ' | ETA: ' + self.get_time(self.eta)
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        self.close()

    def close(self):
        if self.i >= self.max_steps:
            self.total_time = self.elapsed_time()
            print('\nTotal time elapsed: ' + self.get_time(self.total_time))


class Fake_Label(object):
    def __init__(self, model_name):
        label_txt = 'fake_label_{}.txt'.format(model_name)
        self.label_name = os.path.abspath(label_txt)
        with open(self.label_name, 'w') as fd:
            fd.write(' \n \n \n \n \n \n')

    def __del__(self):
        if os.path.exists(self.label_name):
            os.remove(self.label_name)


def find_path(path, name):
    if os.path.basename(path) == name:
        return path
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.abspath(os.path.join(root, name))
    print('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))
    raise FileNotFoundError('File `{}` not found in directory `{}`'.format(name, os.path.abspath(path)))


def subsets_generate(image_list, num_subsets):
    dataset_size = len(image_list)
    if dataset_size < num_subsets:
        yield image_list
    else:
        left_data = dataset_size % num_subsets
        if left_data != 0:
            batch_size = dataset_size // num_subsets + 1
        else:
            batch_size = dataset_size // num_subsets
        for i in range(batch_size):
            if i != batch_size - 1:
                yield image_list[i * num_subsets: (i + 1) * num_subsets]
            else:
                yield image_list[i * num_subsets:]


class Move_Log(object):
    def __init__(self, clean=True):
        self.clean = clean

    def __del__(self):
        if self.clean:
            renew_folder('log')
            txts = ['tensor_min_max.txt', 'tensor_qab.txt', 'tensor_statistics.txt', 'tensor_statistics_type.txt',
                    'softmax_lut_range.txt', 'tensor_weight.txt', 'tensor_weight_calibration.txt', 'tensor_chn.txt',
                    'llvm_record.dat']
            logs = ['statistics.log', 'simulator.log', 'offline.log', 'float.log']
            if os.path.exists('output'):
                shutil.move('output', 'log')
            if os.path.exists('statistics'):
                shutil.move('statistics', 'log')
            if os.path.exists('tmp_image'):
                shutil.move('tmp_image', 'log')
            if os.path.exists('statistics'):
                shutil.move('statistics', 'log')
            file_list = os.listdir(os.getcwd())
            for item in file_list:
                if os.path.basename(item) in txts:
                    shutil.move(item, 'log')
                if item.split('_')[-1] in logs:
                    shutil.move(item, 'log')


def renew_folder(folder_name):
    if os.path.exists(folder_name):
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        else:
            os.remove(folder_name)
            os.mkdir(folder_name)
    else:
        os.mkdir(folder_name)


def print_model(model):
    main_str = 'Net:\n'
    main_str += 'model ' + calibrator_custom.SIM_Simulator.print_model(model)
    printf(main_str)


def write_txt(f, data):
    data_list = data.flatten().tolist()
    for num, value in enumerate(data_list):
        f.write('{:.6f}  '.format(value))
        if (num + 1) % 16 == 0:
            f.write('\n')
    if len(data_list) % 16 != 0:
        f.write('\n')


def random_color(num=100):
    colors = np.random.randint(0, 255, (num, 1, 3))
    colors_100 = []
    for i in range(num):
        colors_100.append(list(colors[i][0].flat))
    return colors_100


def postDetection(model_path, img_path, result_list, out_details, draw_result, show_log=False):
    im = img_path if not isinstance(img_path, list) else img_path[0]
    txt_name = './output/detection_{}_{}.txt'.format(
        os.path.basename(model_path), os.path.basename(im))
    for idx, result in enumerate(result_list):
        if result.shape[-1] != out_details[idx]['shape'][-1]:
            result_list[idx] = result[..., :out_details[idx]['shape'][-1]]
    coordinate = result_list[0]
    img_tmp = cv2.imread(im)
    coordinate[:, :, 0::2] *= img_tmp.shape[0]
    coordinate[:, :, 1::2] *= img_tmp.shape[1]
    list_dict = []
    for i in range(int(result_list[3][0])):
        rr_dict = {}
        rr_dict['image_id'] = os.path.basename(im).split('.')[0]
        rr_dict['image_id'] = int(rr_dict['image_id']) if rr_dict['image_id'].isdigit() else 0
        rr_dict['category_id'] = int(result_list[1][:, i][0]) + 1
        rr_dict['bbox'] = [result_list[0][:, i, :][0][1], result_list[0][:, i, :][0][0],
                           result_list[0][:, i, :][0][3] - result_list[0][:, i, :][0][1],
                           result_list[0][:, i, :][0][2] - result_list[0][:, i, :][0][0]]
        rr_dict['score'] = result_list[2][:, i][0]
        if len(result_list) > 4:
            rr_dict['index'] = int(result_list[4][:, i][0])
        list_dict.append(rr_dict)

    if show_log:
        printf(os.path.basename(im))
    with open(txt_name, 'w') as f:
        for bbox in list_dict:
            if len(result_list) > 4:
                if show_log:
                    printf('{{\"image_id\": {}, \"category_id\": {}, \"bbox\": [{:.6f},{:.6f},{:.6f},{:.6f}], \"score\": {:.6f}, \"index\": {}}},'.format(
                        bbox['image_id'], bbox['category_id'], bbox['bbox'][0], bbox['bbox'][1],
                        bbox['bbox'][2], bbox['bbox'][3], bbox['score'], rr_dict['index']
                    ))
                f.write('{{\"image_id\": {}, \"category_id\": {}, \"bbox\": [{:.6f},{:.6f},{:.6f},{:.6f}], \"score\": {:.6f}, \"index\": {}}},\n'.format(
                    bbox['image_id'], bbox['category_id'], bbox['bbox'][0], bbox['bbox'][1],
                    bbox['bbox'][2], bbox['bbox'][3], bbox['score'], rr_dict['index']
                ))
            else:
                if show_log:
                    printf('{{\"image_id\": {}, \"category_id\": {}, \"bbox\": [{:.6f},{:.6f},{:.6f},{:.6f}], \"score\": {:.6f}}},'.format(
                        bbox['image_id'], bbox['category_id'], bbox['bbox'][0], bbox['bbox'][1],
                        bbox['bbox'][2], bbox['bbox'][3], bbox['score']
                    ))
                f.write('{{\"image_id\": {}, \"category_id\": {}, \"bbox\": [{:.6f},{:.6f},{:.6f},{:.6f}], \"score\": {:.6f}}},\n'.format(
                    bbox['image_id'], bbox['category_id'], bbox['bbox'][0], bbox['bbox'][1],
                    bbox['bbox'][2], bbox['bbox'][3], bbox['score']
                ))

    if draw_result is not None:
        out_result = draw_result.split(',')
        threshold = 0
        colors = random_color()
        if len(out_result) > 1:
            threshold = float(out_result[1])
        for bbox in list_dict:
            if bbox['score'] > threshold:
                B = int(colors[bbox['category_id']][0])
                G = int(colors[bbox['category_id']][1])
                R = int(colors[bbox['category_id']][2])
                cv2.rectangle(img_tmp, (int(bbox['bbox'][0]), int(bbox['bbox'][1]), int(bbox['bbox'][2]),
                            int(bbox['bbox'][3])), (B, G, R), 4)
                text = '{}: {:.2f}'.format(bbox['category_id'], bbox['score'])
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_tmp, (int(bbox['bbox'][0]), int(bbox['bbox'][1])), (int(bbox['bbox'][0]) + text_size[0][0],
                            int(bbox['bbox'][1]) + text_size[0][1]), (B, G, R), -1)
                cv2.putText(img_tmp, text,
                        (int(bbox['bbox'][0]), int(bbox['bbox'][1]) + text_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, 8)
        if not os.path.exists(out_result[0]):
            os.mkdir(out_result[0])
        out_result[0] = os.path.join(out_result[0], os.path.basename(im))
        cv2.imwrite(out_result[0], img_tmp)


def postClassification(model_path, img_path, result_list, out_details, show_log=False):
    im = img_path if not isinstance(img_path, list) else img_path[0]
    result = result_list[0]
    if result.shape[-1] != out_details[0]['shape'][-1]:
        result = result[..., :out_details[0]['shape'][-1]]
    out_result = result.reshape(-1)
    ordered = np.argsort(-out_result)
    out = [(i, out_result[i]) for i in ordered[:5]]
    txt_name = './output/classification_{}_{}.txt'.format(
        os.path.basename(model_path), os.path.basename(im))
    with open(txt_name, 'w') as f:
        if show_log:
            printf(os.path.basename(im))
        f.write(os.path.basename(im))
        f.write('\n')
        for order, (index, prob) in enumerate(out):
            if show_log:
                printf('Order: {} index: {} {:.6f}'.format(order + 1, index, prob))
            f.write('{} {:.6f}\n'.format(index, prob))


def postUnknown(model_path, img_path, result_list, out_details, skip_garbage):
    im = img_path if not isinstance(img_path, list) else img_path[0]
    txt_name = './output/unknown_{}_{}.txt'.format(
        os.path.basename(model_path), os.path.basename(im))
    with open(txt_name, 'w') as f:
        for i, result in enumerate(result_list):
            f.write('{} Tensor:\n{{\n'.format(out_details[i]['name']))
            if 'quantization' in out_details[i] and not skip_garbage:
                f.write('tensor dim:{},  Original shape:[{}],  Alignment shape:[{}]\n'
                    .format(out_details[i]['shape'].shape[0], ' '.join(str(v) for v in list(out_details[i]['shape'].flat)),
                    ' '.join(str(v) for v in list(result.shape))))
                f.write('The following tensor data shape is alignment shape.\n')
            else:
                if result.shape[-1] != out_details[i]['shape'][-1]:
                    result = result[..., :out_details[i]['shape'][-1]]
                f.write('tensor dim:{},  Original shape:[{}]\n'
                    .format(out_details[i]['shape'].shape[0], ' '.join(str(v) for v in list(out_details[i]['shape'].flat))))
            f.write('tensor data:\n')
            write_txt(f, result)
            f.write('}\n\n')


def invoke_model(x, model, model_path, img_path, category, dump_rawdata, draw_result, skip_garbage, show_log=False):
    in_details = model.get_input_details()
    out_details = model.get_output_details()
    for idx, _ in enumerate(in_details):
        if dump_rawdata:
            if isinstance(img_path, list):
                utils.convert_to_input_formats(x[idx], in_details[idx]).tofile('{}_input{}.bin'.format(os.path.basename(img_path[idx]), idx))
                printf("Input{} rawdata saved in {}_input{}.bin".format(idx, os.path.basename(img_path[idx]), idx))
            else:
                utils.convert_to_input_formats(x[idx], in_details[idx]).tofile('{}.bin'.format(os.path.basename(img_path)))
                printf("Input rawdata saved in {}.bin".format(os.path.basename(img_path)))
        model.set_input(idx, utils.convert_to_input_formats(x[idx], in_details[idx]))
    model.invoke()
    result_list = []
    for idx, _ in enumerate(out_details):
        result = model.get_output(idx)
        # for Fixed and Offline model
        if out_details[idx]['dtype'] == np.int16:
            scale, _ = out_details[idx]['quantization']
            result = np.multiply(result, scale)
        result_list.append(result)
    if category == 'Detection':
        if model.class_type != 'OFFLINE' and not model.is_detection():
            raise ValueError('Model doesn\'t concat SGS Postprocess method, can\'t handle Detection!')
        postDetection(model_path, img_path, result_list, out_details, draw_result, show_log=show_log)
    elif category == 'Classification':
        postClassification(model_path, img_path, result_list, out_details, show_log=show_log)
    else:
        postUnknown(model_path, img_path, result_list, out_details, skip_garbage)


def cal_simulator_multi(image_list, phase, model_path, category, preprocess_func, dump_rawdata,
                         draw_result, skip_garbage, input_config, num_subsets=10, subset_index=0):
    image_size = len(image_list)
    subset_size = list(((image_size // num_subsets) * np.ones((1, num_subsets), dtype=np.int64)).flat)
    left_size = image_size % num_subsets
    if (left_size != 0):
        for i_ in range(len(subset_size)):
            if (i_ // left_size == 0):
                subset_size[i_] += 1
    bShow = False if num_subsets > 1 else True
    log = SHOW_LOG or bShow
    if subset_index == 0 and not bShow:
        process_bar = ShowProcess(subset_size[0])

    if num_subsets != subset_index:
        id_start = 0
        if subset_index > 0:
            for i_, size_ in enumerate(subset_size):
                if i_ < subset_index:
                    id_start += size_
        id_end = id_start + subset_size[subset_index]

        if phase == 'Float':
            model = calibrator_custom.float_simulator(model_path, show_log=log)
        elif phase == 'Fixed':
            model = calibrator_custom.fixed_simulator(model_path, show_log=log)
        elif phase == 'Cmodel_float':
            model = calibrator_custom.cmodel_float_simulator(model_path, show_log=log)
        elif phase == 'Fixed_without_ipu_ctrl':
            model = calibrator_custom.fixed_wipu_simulator(model_path, show_log=log)
        else:
            model = calibrator_custom.offline_simulator(model_path, input_config=input_config, show_log=log)

        if subset_index == 0:
            print_model(model)
        norm = True if model.class_type in ['FLOAT', 'CMODEL_FLOAT'] else False

        for i in range(id_start, id_end):
            im_file = image_list[i]
            if len(preprocess_func) > 1:
                if len(preprocess_func) != len(im_file):
                    raise ValueError('Got different num of preprocess_methods and images!')
                output_img = []
                for idx, preprocess in enumerate(preprocess_func):
                    output_img.append(preprocess(im_file[idx], norm=norm))
            else:
                if isinstance(im_file, list):
                    if len(preprocess_func) != 1 or len(im_file) != 1:
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = [preprocess_func[0](im_file[0], norm=norm)]
                else:
                    output_img = [preprocess_func[0](im_file, norm=norm)]
            gc.collect()
            invoke_model(output_img, model, model_path, im_file, category, dump_rawdata, draw_result,
                         skip_garbage, show_log=bShow)
            if subset_index == 0 and not bShow:
                process_bar.show_process()


def run_simulator(base_name, image_list, phase, model_path, category, preprocess_func, num_subsets,
                      dump_rawdata, draw_result, skip_garbage, input_config):
    if len(image_list) == 0:
        print('\033[33mThe input data is empty, please check the input parameters!\033[0m')
        raise ValueError('The input data is empty, please check the input parameters!')
    if len(image_list) < num_subsets:
        num_subsets = len(image_list)
    if num_subsets > 1:
        printf('\033[31mStart to evaluate on {}...\033[0m'.format(base_name.strip().split('/')[-2]
            if base_name.strip().split('/')[-1] == '' else base_name.strip().split('/')[-1]))
        p_list = []
        p_simulator = Pool(processes=os.cpu_count())
        for im_ss in range(num_subsets):
            p_list.append(p_simulator.apply_async(cal_simulator_multi, args=(image_list, phase,
                model_path, category, preprocess_func, dump_rawdata, draw_result, skip_garbage, input_config),
                kwds={'num_subsets': num_subsets, 'subset_index': im_ss}))
        p_simulator.close()
        p_simulator.join()
        if p_list[0].get() is not None:
            print(p_list[0].get())
        else:
            printf('\033[31mRun evaluation OK.\033[0m')
    else:
        cal_simulator_multi(image_list, phase, model_path, category,
            preprocess_func, dump_rawdata, draw_result, skip_garbage, input_config, num_subsets=1, subset_index=0)