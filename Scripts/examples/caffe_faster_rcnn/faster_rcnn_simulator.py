# -*- coding: utf-8 -*-

import calibrator_custom
import cv2
import os
import rpn
import copy
import numpy as np
import argparse
import pickle
from calibrator_custom import utils


def arg_parse():
    parser = argparse.ArgumentParser(description='Faster RCNN Simulator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path.')
    parser.add_argument('-m0', '--model0', type=str, required=True,
                        help='Main Model path.')
    parser.add_argument('-m1', '--model1', type=str, required=True,
                        help='Second Model path.')
    parser.add_argument('-t', '--type', type=str, required=True,
                        choices=['Float', 'Fixed', 'Offline'],
                        help='Indicate model data type.')
    parser.add_argument('--num_process', default=1, type=int, help='Amount of processes run at same time.')

    return parser.parse_args()

def check_input_img(img_path):
    img = cv2.imread(img_path)
    width = img.shape[1]
    height = img.shape[0]
    h_w = height / width
    like_4_3 = lambda x: x + 3 - (x * 4 % 3)
    like_3_4 = lambda x: x + 4 - (x * 3 % 4)
    if h_w > 0.75:
        h = like_4_3(height)
        w = h * 4 // 3
        img_out = np.pad(img, ([0, h-height], [0, w-width], [0, 0]), 'constant')
        print('The picture: %s ratio is not 4: 3, the filled size is [H: %d, W: %d]' % (img_path, img_out.shape[0], img_out.shape[1]))
    elif h_w < 0.75:
        w = like_3_4(width)
        h = w * 3 // 4
        img_out = np.pad(img, ([0, h-height], [0, w-width], [0, 0]), 'constant')
        print('The picture: %s ratio is not 4: 3, the filled size is [H: %d, W: %d]' % (img_path, img_out.shape[0], img_out.shape[1]))
    else:
        img_out = img

    return img_out

def image_generator(image_list, preprocess_func):
    for image in image_list:
        img = preprocess_func(image)
        yield [img]

def fill_inputImg2main(img_out, resizeH=600, resizeW=800, resizeC=3, norm=True, meanB=102.9801, meanG=115.9465, meanR=122.7717, std=1, nchw=False):
    width = img_out.shape[1]
    height = img_out.shape[0]
    im_max = max(width, height)
    im_min = min(width, height)
    im_scale = resizeH / im_min
    if (im_scale * im_max) > resizeW:
        im_scale = resizeW / im_max

    img_norm = cv2.resize(img_out, (0, 0), fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    if norm:
        img_norm = (img_norm - [meanB, meanG, meanR]) / std
        img_norm = img_norm.astype('float32')

    if nchw:
        # NCHW
        img_norm = np.transpose(img_norm.reshape(resizeW, resizeH, -1), axes=(2, 0, 1))

    return np.expand_dims(img_norm, 0), im_scale

def visualize_bbox(im, class_name, color, dets, save_file, thresh=0.8):
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = [int(num) for num in dets[i, :4]]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
            text_str = '%s: %.3f' % (class_name, score)
            text = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(im, (bbox[0], bbox[3]), (bbox[0] + text[0][0], bbox[3] - text[0][1]), color, -1)
            cv2.putText(im, text_str, (bbox[0], bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, 8)
            print('{} -> ({}, {}) ({}, {})'.format(text_str, bbox[0], bbox[1], bbox[2], bbox[3]))
    cv2.imwrite('out_' + os.path.basename(save_file), im)
    return im

class Net(calibrator_custom.SIM_Simulator):
    def __init__(self, main_model_path, second_model_path, phase):
        super().__init__()
        if phase == 'Float':
            self.main_model = calibrator_custom.float_simulator(main_model_path)
            self.second_model = calibrator_custom.float_simulator(second_model_path)
            self.norm = True
        elif phase == 'Fixed':
            self.main_model = calibrator_custom.fixed_simulator(main_model_path)
            self.second_model = calibrator_custom.fixed_simulator(second_model_path)
            self.norm = False
        else:
            self.main_model = calibrator_custom.offline_simulator(main_model_path)
            self.second_model = calibrator_custom.offline_simulator(second_model_path)
            self.norm = False
        self.rpn = rpn.ProposalLayer()

    def forward(self, x):
        # Run main model
        out_details = self.main_model.get_output_details()
        input_data, im_scale = fill_inputImg2main(x, norm=self.norm)
        self.main_model.set_input(0, input_data)
        self.main_model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.main_model.get_output(idx)
            # Get main output for Fixed and Offline model
            if result.shape[-1] != out_details[idx]['shape'][-1] and out_details[idx]['name'] != 'conv5_3':
                result = result[..., :out_details[idx]['shape'][-1]]
            if out_details[idx]['dtype'] == np.int16 and out_details[idx]['name'] != 'conv5_3':
                scale, _ = out_details[idx]['quantization']
                result = np.dot(result, scale)
            result_list.append(result)
        # Run Proposal Layer
        im_info = np.array([x.shape[0], x.shape[1], im_scale]).reshape(1, 3)
        bottom = [result_list[0], result_list[1], im_info]
        roi = self.rpn.forward(bottom)
        # Run Second stage model
        in2_details = self.second_model.get_input_details()
        out2_details = self.second_model.get_output_details()
        in2_data = [result_list[2], copy.deepcopy(roi)]
        for idx, in2_info in enumerate(in2_details):
            if in2_info['dtype'] == np.int16 and in2_info['name'] != 'conv5_3':
                # Set second model input for Fixed and Offline model
                ins, zp = in2_info['quantization']
                in2_data[idx] = np.clip((in2_data[idx] / ins + zp), -32767, 32767).astype(in2_info['dtype'])
                feature_s16 = np.zeros(in2_info['shape']).astype('int16')
                feature_s16[..., :in2_data[idx].shape[-1]] = in2_data[idx]
                self.second_model.set_input(in2_info['index'], feature_s16)
            else:
                self.second_model.set_input(in2_info['index'], in2_data[idx])
        self.second_model.invoke()
        second_result = []
        for idx, _ in enumerate(out2_details):
            result = self.second_model.get_output(idx)
            # Get second output for Fixed and Offline model
            if result.shape[-1] != out2_details[idx]['shape'][-1]:
                result = result[..., :out2_details[idx]['shape'][-1]]
            if out2_details[idx]['dtype'] == np.int16:
                scale, _ = out2_details[idx]['quantization']
                result = np.dot(result, scale)
            second_result.append(result.reshape(300, -1))
        second_result.append(roi)
        second_result.append(im_info)
        return second_result

def main():
    args = arg_parse()
    image_path = args.image
    main_model_path = args.model0
    second_model_path = args.model1
    phase = args.type
    num_subsets = args.num_process

    if not os.path.exists(main_model_path):
        raise FileNotFoundError('No such {} model'.format(main_model_path))
    if not os.path.exists(second_model_path):
        raise FileNotFoundError('No such {} model'.format(second_model_path))

    net = Net(main_model_path, second_model_path, phase)
    print(net)

    if not os.path.exists(image_path):
        raise FileNotFoundError('No such {} image or directory.'.format(image_path))

    image_list = []
    if os.path.isdir(image_path):
        image_list = [os.path.join(image_path, i) for i in os.listdir(image_path)]
        img_gen = image_generator(image_list, check_input_img)
    else:
        image_list.append(image_path)
        img_gen = image_generator(image_list, check_input_img)

    results = net(img_gen, num_process=num_subsets)

    classes_name = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor']
    random_colors = [[int(j) for j in i.flatten()] for i in np.random.randint(0, 255, (len(classes_name), 1, 3))]

    # Faster RCNN postprocess
    for i, result in enumerate(results):
        img = check_input_img(image_list[i])
        print(image_list[i])
        rois = result[-2]
        # unscale back to raw image space
        boxes = rois[:, 1:5] / result[-1][-1][-1]
        # use softmax estimated probabilities
        scores = result[0]
        # Apply bounding-box regression deltas
        box_deltas = result[1]
        pred_boxes = rpn.bbox_transform_inv(boxes, box_deltas)
        pred_boxes = rpn.clip_boxes(pred_boxes, result[-1][-1][:2])
        for cls_ind, cls in enumerate(classes_name[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = pred_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = rpn.nms(dets, thresh=0.3)
            dets = dets[keep, :]
            img = visualize_bbox(img, cls, random_colors[cls_ind], dets, image_list[i], thresh=0.5)


if __name__ == '__main__':
    main()
