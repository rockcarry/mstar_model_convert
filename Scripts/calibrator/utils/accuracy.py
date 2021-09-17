# -*- coding: utf-8 -*-

import os
import cv2
import shutil
from multiprocessing import Pool
from . import misc
import json
import numpy as np


def which_simulator(tool_path, phase):
    if phase == 'Float':
        label_image_path = misc.find_path(tool_path, 'sgs_simulator_float')
    else:
        label_image_path = misc.find_path(tool_path, 'sgs_simulator_fixed')

    return label_image_path


def label_image_cmd(tool_path, image, label, model, category, skip_garbage, phase, debug=False):
    if debug:
        debug_info = 'gdb --args '
    else:
        debug_info = ''
    if phase == 'Offline':
        label_cmd = '{}{} -i \'{}\' -l {} -m {} -c {} -d offline --skip_preprocess {}'.format(debug_info, tool_path, image, label, model, category, skip_garbage)
    else:
        label_cmd = '{}{} -i \'{}\' -l {} -m {} -c {} --skip_preprocess {}'.format(debug_info, tool_path, image, label, model, category, skip_garbage)

    return label_cmd


def output_result(category, model, image):
    if category == 'Classification':
        out_txt = 'classification_{}_{}.txt'.format(os.path.basename(model), image.strip().split('/')[-1])
    elif category == 'Detection':
        out_txt = 'detection_{}_{}.txt'.format(os.path.basename(model), image.strip().split('/')[-1])
    elif category == 'Unknown':
        out_txt = 'unknown_{}_{}.txt'.format(os.path.basename(model), image.strip().split('/')[-1])

    return out_txt


def label_image(image_list, model, label, category, model_name, preprocess_func, project_path, label_image_path, phase, draw, skip_garbage, continue_run, debug=False, save_in=False):
    output_dir = 'output'
    if not continue_run:
        misc.renew_folder(output_dir)
    tmp_image_dir = 'tmp_image'
    misc.renew_folder(tmp_image_dir)
    for im in image_list:
        if model_name != '0':
            if len(preprocess_func) > 1:
                if len(preprocess_func) != len(im):
                    raise ValueError('Got different num of preprocess_methods and images!')
                output_img = ''
                for idx, preprocess in enumerate(preprocess_func):
                    out_img = os.path.join(tmp_image_dir, '{}_{}'.format(idx, os.path.basename(im[idx])))
                    if phase == 'Float':
                        misc.convert_image(im[idx], out_img, preprocess)
                    else:
                        misc.convert_image(im[idx], out_img, preprocess, norm=False)
                    output_img += out_img
                    if idx < len(preprocess_func) - 1:
                        output_img += ':'
            else:
                if isinstance(im, list):
                    if len(preprocess_func) != 1 or len(im) != 1:
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = os.path.join('tmp_image', '{}'.format(os.path.basename(im[0])))
                    if phase == 'Float':
                        misc.convert_image(im[0], output_img, preprocess_func[0])
                    else:
                        misc.convert_image(im[0], output_img, preprocess_func[0], norm=False)
                else:
                    output_img = os.path.join(tmp_image_dir, '{}'.format(os.path.basename(im)))
                    if phase == 'Float':
                        misc.convert_image(im, output_img, preprocess_func[0])
                    else:
                        misc.convert_image(im, output_img, preprocess_func[0], norm=False)
            out_result = os.path.join(output_dir, output_result(category, model, output_img))
            label_cmd = label_image_cmd(label_image_path, output_img, label, model, category, skip_garbage, phase, debug)
            if debug:
                print('\033[33m================Debug command================\033[0m\n' + label_cmd + '\n\033[33m=============================================\033[0m')
            os.system(label_cmd)
            if not os.path.exists(out_result):
                raise RuntimeError('Run simulator failed!\nUse command to debug: {}'.format(label_cmd))
            if not save_in:
                for out_img in output_img.split(':'):
                    os.remove(out_img)
            if (draw != '0') and (category == 'Detection'):
                colors = random_color()
                draw_rectangle(im, out_result, project_path, draw, colors)
        else:
            label_cmd = label_image_cmd(label_image_path, im, label, model, category, skip_garbage, phase, debug)
            if debug:
                print('\033[33m================Debug command================\033[0m\n' + label_cmd + '\n\033[33m=============================================\033[0m')
            os.system(label_cmd)
            out_result = os.path.join(output_dir, output_result(category, model, im))
            if not os.path.exists(out_result):
                raise RuntimeError('Run simulator failed!\nUse command to debug: {}'.format(label_cmd))
            if (draw != '0') and (category == 'Detection'):
                colors = random_color()
                draw_rectangle(im, out_result, project_path, draw, colors)
    if not save_in:
        shutil.rmtree(tmp_image_dir)


def cal_simulator_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path,
                        fake_label, skip_garbage, phase, dataset_size=1, save_in=False, num_subsets=10, subset_index=0):
    image_size = len(image_list)
    subset_size = list(((image_size // num_subsets) * np.ones((1, num_subsets), dtype=np.int)).flat)
    left_size = image_size % num_subsets
    if (left_size != 0):
        for i_ in range(len(subset_size)):
            if (i_ // left_size == 0):
                subset_size[i_] += 1
    if subset_index == 0:
        process_bar = misc.ShowProcess(subset_size[0])

    if num_subsets == subset_index:
        if category == 'Classification':
            image_label = dict()
            with open(label, 'r') as f:
                lines = f.readlines()
            for line in lines:
                label_line = line.strip().split(' ')
                image_label[label_line[0].strip().split('.')[-2]] = int(label_line[1])
            count_top1 = 0
            count_top5 = 0
            results_list = os.listdir('output')
            prefix_result = '{}_{}_'.format(category, os.path.basename(model))
            results_list = [os.path.join('output', i) for i in results_list]
            for im_file in results_list:
                with open(im_file, 'r') as f:
                    results = f.readlines()
                results_top5 = [int(results[i].strip().split(' ')[0]) for i in range(1, len(results))]
                im_label = image_label[im_file.split('/')[-1][len(prefix_result): -4].split('.')[-2]]
                if im_label in results_top5:
                    count_top5 += 1
                    if im_label == results_top5[0]:
                        count_top1 += 1
            top1_acc = count_top1 / dataset_size
            top5_acc = count_top5 / dataset_size
            classify_acc = '\x0A\x0D{} images\x0A\x0Dtop1: {:.2%}\x0A\x0Dtop5: {:.2%}\x0A\x0D'.format(dataset_size, top1_acc, top5_acc)
            print(classify_acc)
            classify_txt = '{}_{}_acc.log'.format(os.path.basename(model), dataset_size)
            with open(classify_txt, 'w') as f:
                f.write(classify_acc)
        elif category == 'Detection':
            merged_results_json = os.path.join('output', 'detection_{}_merge.json'.format(os.path.basename(model).split('.')[0]))
            results_list = os.listdir('output')
            prefix_result = '{}_{}_'.format(category, os.path.basename(model))
            results_list = [os.path.join('output', i) for i in results_list if i.split('.')[-1] == 'txt']
            with open(merged_results_json, 'w') as fw:
                fw.write('[')
                for im_file in results_list:
                    with open(im_file, 'r') as fr:
                        lines = fr.readlines()
                    contents = ''.join(lines)
                    if im_file == results_list[-1]:
                        fw.write(contents[:-2])
                    else:
                        fw.write(contents)
                fw.write(']')

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            coco_gt = COCO(label)
            coco_dt = coco_gt.loadRes(merged_results_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            detection_txt = '{}_{}_mAP.log'.format(os.path.basename(model), dataset_size)
            with open(detection_txt, 'w') as f:
                f.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[0]))
                f.write('Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[1]))
                f.write('Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[2]))
                f.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[3]))
                f.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[4]))
                f.write('Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[5]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}\n'.format(coco_eval.stats[6]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n'.format(coco_eval.stats[7]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[8]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[9]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[10]))
                f.write('Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n'.format(coco_eval.stats[11]))

        else:
            raise ValueError('Not support {} category for evaluate datasets.'.format(category))

    else:
        id_start = 0
        if subset_index > 0:
            for i_, size_ in enumerate(subset_size):
                if i_ < subset_index:
                    id_start += size_
        id_end = id_start + subset_size[subset_index]

        for i in range(id_start, id_end):
            im_file = image_list[i]
            if model_name != '0':
                if len(preprocess_func) > 1:
                    if len(preprocess_func) != len(im_file):
                        raise ValueError('Got different num of preprocess_methods and images!')
                    output_img = ''
                    for idx, preprocess in enumerate(preprocess_func):
                        out_img = os.path.join('tmp_image', '{}_{}'.format(idx, os.path.basename(im_file[idx])))
                        if phase == 'Float':
                            misc.convert_image(im_file[idx], out_img, preprocess)
                        else:
                            misc.convert_image(im_file[idx], out_img, preprocess, norm=False)
                        output_img += out_img
                        if idx < len(preprocess_func) - 1:
                            output_img += ':'
                else:
                    if isinstance(im_file, list):
                        if len(preprocess_func) != 1 or len(im_file) != 1:
                            raise ValueError('Got different num of preprocess_methods and images!')
                        output_img = os.path.join('tmp_image', '{}'.format(os.path.basename(im_file[0])))
                        if phase == 'Float':
                            misc.convert_image(im_file[0], output_img, preprocess_func[0])
                        else:
                            misc.convert_image(im_file[0], output_img, preprocess_func[0], norm=False)
                    else:
                        output_img = os.path.join('tmp_image', '{}'.format(os.path.basename(im_file)))
                        if phase == 'Float':
                            misc.convert_image(im_file, output_img, preprocess_func[0])
                        else:
                            misc.convert_image(im_file, output_img, preprocess_func[0], norm=False)
                out_result = os.path.join('output', output_result(category, model, output_img))
                label_cmd = label_image_cmd(tool_path, output_img, fake_label, model, category, skip_garbage, phase)
                label_cmd = label_cmd + ' >> {}_simulator.log'.format(model.split('.')[-2].split('/')[-1])
                os.system(label_cmd)
                if subset_index == 0:
                    process_bar.show_process()
                if not os.path.exists(out_result):
                    raise RuntimeError('Run simulator failed!\nUse command to debug: {}'.format(label_cmd))
                if not save_in:
                    for out_img in output_img.split(':'):
                        os.remove(out_img)

            else:
                out_result = os.path.join('output', output_result(category, model, im_file))
                label_cmd = label_image_cmd(tool_path, im_file, fake_label, model, category, skip_garbage, phase)
                label_cmd = label_cmd + ' >> {}_simulator.log'.format(model.split('.')[-2].split('/')[-1])
                os.system(label_cmd)
                if subset_index == 0:
                    process_bar.show_process()
                if not os.path.exists(out_result):
                    raise RuntimeError('Run simulator failed!\nUse command to debug: {}'.format(label_cmd))

def run_simulator(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path,
                  phase, skip_garbage, continue_run, dataset_size, save_in=False, num_subsets=10):
    output_dir = os.path.join('output')
    if not continue_run:
        misc.renew_folder(output_dir)
    tmp_image_dir = 'tmp_image'
    misc.renew_folder(tmp_image_dir)
    if len(image_list) < num_subsets:
        num_subsets = len(image_list)
    p_simulator = Pool(processes=os.cpu_count())
    for im_ss in range(num_subsets):
        p_simulator.apply_async(cal_simulator_multi, args=(image_list, label, model, category, model_name, preprocess_func,
                                project_path, tool_path, label, skip_garbage, phase), kwds={'save_in': save_in,
                                'num_subsets': num_subsets, 'subset_index': im_ss})
    p_simulator.close()
    p_simulator.join()
    if (category != "Unknown"):
        cal_simulator_multi(image_list, label, model, category, model_name, preprocess_func, project_path, tool_path, label,
                            skip_garbage, phase, dataset_size, save_in, subset_index=10)
    if not save_in:
        shutil.rmtree(tmp_image_dir)


def convert_json(output_txt, project_path):
    output_json = os.path.join('tmp_image', '{}.json'.format(output_txt.strip().split('/')[-1]))
    fw = open(output_json, 'w')
    fw.write('[')
    with open(output_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == lines[-1]:
                fw.write(line[:-2])
            else:
                fw.write(line)
    fw.write(']')
    fw.close()
    return json.load(open(output_json))


def random_color():
    colors = np.random.randint(0, 255, (100, 1, 3))
    colors_100 = []
    for i in range(100):
        colors_100.append(list(colors[i][0].flat))
    return colors_100


def draw_rectangle(im_file, output_txt, project_path, output_result, colors):
    im = cv2.imread(im_file, flags=-1)
    bbox = convert_json(output_txt, project_path)
    out_result = output_result.split(',')
    threshold = 0
    if len(out_result) > 1:
        threshold = float(out_result[1])
    for i in range(len(bbox)):
        if bbox[i]['score'] > threshold:
            B = int(colors[bbox[i]['category_id']][0])
            G = int(colors[bbox[i]['category_id']][1])
            R = int(colors[bbox[i]['category_id']][2])
            cv2.rectangle(im, (int(bbox[i]['bbox'][0]), int(bbox[i]['bbox'][1]), int(bbox[i]['bbox'][2]),
                          int(bbox[i]['bbox'][3])), (B, G, R), 2)
            cv2.putText(im, '{}:{:.2f}'.format(bbox[i]['category_id'], bbox[i]['score']),
                       (int(bbox[i]['bbox'][0]), int(bbox[i]['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (B, G, R), 1)
    if not os.path.exists(out_result[0]):
        os.mkdir(out_result[0])
    out_result[0] = os.path.join(out_result[0], im_file.strip().split('/')[-1])
    cv2.imwrite(out_result[0], im)
