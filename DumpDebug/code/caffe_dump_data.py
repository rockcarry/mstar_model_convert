# -- coding: UTF-8 --
import sys
caffe_root = '/data2/eason/V7.7/Sigmastar_Devkit/src/caffe'
sys.path.insert(0,caffe_root+'/python')

import argparse
import caffe
import numpy as np
import cv2
import scipy.io as spio
import os
import pdb
import struct
import importlib
import sys
import time
import re
import shutil
import subprocess

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

def image_preprocess_func(model_name):
    if os.path.exists(model_name) and model_name.split('.')[-1] == 'py':
        sys.path.append(os.path.dirname(model_name))
        preprocess_func = importlib.import_module(os.path.basename(model_name).split('.')[0])
    else:
        raise Exception('preprocess function {} not found'.format(model_name))
    return preprocess_func.get_image

def image_generator(image_list, preprocess_funcs, norm=True):
    for image in image_list:
        imgs = []
        if len(preprocess_funcs) > 1:
            if len(preprocess_funcs) != len(image):
                raise ValueError('Got different num of preprocess_methods and images!')
            for idx, preprocess_func in enumerate(preprocess_funcs):
                imgs.append(preprocess_func(image[idx], norm))
            yield [imgs]
        else:
            if isinstance(image, list):
                if len(preprocess_funcs) != 1 or len(image) != 1:
                    raise ValueError('Got different num of preprocess_methods and images!')
                imgs.append(preprocess_funcs[0](image[0], norm))
                yield [imgs]
            else:
                imgs.append(preprocess_funcs[0](image, norm))
                yield [imgs]

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.basename(apath).split('.')[-1].lower() in image_suffix:
                result.append(os.path.abspath(apath))
    if len(result) == 0:
        raise ValueError('No images found in {}'.format(dirname))
    return result

image_suffix = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm',
                'sr', 'ras', 'tiff', 'tif', 'hdr', 'pic', 'raw', 'npy', 'bin', 'mp4']


def extract_feature(net, image_path, preprocess_funcs, dump_bin):
    # load image
    # inputImage = get_image(image_path)

    for i in range(len(net.inputs)):
        inputImage = preprocess_funcs[i](image_path[i], norm=True, nchw=True)
        net_input = net.inputs[i]
        net.blobs[net_input].reshape(*inputImage.shape)
        net.blobs[net_input].data[...] = inputImage

    #np.set_printoptions(threshold=np.inf)

    #caffe forward
    net.forward()

    shapeString = ''
    for layer_name, blob in net.blobs.iteritems():
      shapeString += layer_name.replace("/","_") + '\t' + str(blob.data.shape) + '\n'

    f = open('dumpData/shape.txt','w')
    f.write(str(shapeString))
    f.close()

    i = 0
    if (dump_bin != "True"):
        for layer_name, blob in net.blobs.iteritems():
          raw_data = blob.data
          layer_name = layer_name.replace("/","_")

          layer_name_file = layer_name.replace("/","_")
          layer_name_file = layer_name_file.replace(":","_")
          layer_name_file = layer_name_file.replace("*","_")
          layer_name_file = layer_name_file.replace("?","_")
          layer_name_file = layer_name_file.replace("<","_")
          layer_name_file = layer_name_file.replace(">","_")
          layer_name_file = layer_name_file.replace("|","_")

          # dump each layer in seperated files as NCHW string
          f = open('dumpData/NCHW/'+layer_name_file + '#' + str(raw_data.shape),'wb')
          head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + layer_name + " bConstant:0 " + "shape:["

          for j in range(len(raw_data.shape)):
              if j == (len(raw_data.shape) - 1):
                  head1 = head1 + str(raw_data.shape[j]) + "] "
              else:
                  head1 = head1 + str(raw_data.shape[j]) + ", "

          head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
          f.write(head1.encode('ascii'))
          #write head2
          head2 = "op_out[" + str(i) + "] " + layer_name + " = {\n"
          f.write(head2.encode('ascii'))
          #write head3
          head3 = "//buffer data size: " + str(raw_data.size*4) + "\n\n"
          f.write(head3.encode('ascii'))

          for num, value in enumerate(list(raw_data.flat)):
              f.write('{:.6f}, '.format(value))
              if (num + 1) % 16 == 0:
                  f.write('\n')
          f.write('\n')
          f.write('}\n')

          f.close

          # dump each layer in seperated files as NHWC string
          if len(raw_data.shape) == 4:
              raw_data = np.transpose(raw_data,[0,2,3,1])

          f = open('dumpData/NHWC/'+layer_name_file + '#' + str(raw_data.shape),'wb')
          head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + layer_name + " bConstant:0 " + "shape:["

          for j in range(len(raw_data.shape)):
              if j == (len(raw_data.shape) - 1):
                  head1 = head1 + str(raw_data.shape[j]) + "] "
              else:
                  head1 = head1 + str(raw_data.shape[j]) + ", "

          head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
          f.write(head1)
          #write head2
          head2 = "op_out[" + str(i) + "] " + layer_name + " = {\n"
          f.write(head2)
          #write head3
          head3 = "//buffer data size: " + str(raw_data.size*4) + "\n\n"
          f.write(head3)


          for num, value in enumerate(list(raw_data.flat)):
              f.write('{:.6f}, '.format(value))
              if (num + 1) % 16 == 0:
                  f.write('\n')
          f.write('\n')
          f.write('}\n')

          i +=1
          f.close

        # dump all layers in one file as NHWC string
        with open(r"dumpData/caffe_NHWC_outtensor_dump.txt","wb") as f:
            head = "isFloat: 1\n"
            #head_w = bytearray(struct.pack("s", head))
            f.write(head)
            i = 0
            for layer_name, blob in net.blobs.iteritems():
                if net.inputs[0] == layer_name:
                    continue
                raw_data = blob.data
                if len(raw_data.shape) == 4:
                    raw_data = np.transpose(raw_data,[0,2,3,1])

                #write head1
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + layer_name + " bConstant:0 " + "shape:["

                for j in range(len(raw_data.shape)):
                    if j == (len(raw_data.shape) - 1):
                        head1 = head1 + str(raw_data.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
                f.write(head1)
                #write head2
                head2 = "op_out[" + str(i) + "] " + layer_name + " = {\n"
                f.write(head2)
                #write head3
                head3 = "//buffer data size: " + str(raw_data.size*4) + "\n\n"
                f.write(head3)

                for num, value in enumerate(list(raw_data.flat)):
                    f.write('{:.6f}, '.format(value))
                    if (num + 1) % 16 == 0:
                        f.write('\n')
                f.write('\n')
                f.write('}\n')
                i +=1
        f.close
    else:
        # dump all layers in one file as NHWC bin
        with open(r"dumpData/caffe_NHWC_outtensor_dump.bin","wb") as f:
            head = "isFloat: 1\n"
            #head_w = bytearray(struct.pack("s", head))
            f.write(head)
            i = 0
            for layer_name, blob in net.blobs.iteritems():
                if net.inputs[0] == layer_name:
                    continue
                raw_data = blob.data
                if len(raw_data.shape) == 4:
                    raw_data = np.transpose(raw_data,[0,2,3,1])

                #write head1
                head1 = "//out 0 s: 0.000000 z: 0 type: float name: " + layer_name + " bConstant:0 " + "shape:["

                for j in range(len(raw_data.shape)):
                    if j == (len(raw_data.shape) - 1):
                        head1 = head1 + str(raw_data.shape[j]) + "] "
                    else:
                        head1 = head1 + str(raw_data.shape[j]) + ", "

                head1 = head1 + "dims:" + str(len(raw_data.shape)) + "\n"
                f.write(head1.encode('ascii'))
                #write head2
                head2 = "op_out[" + str(i) + "] " + layer_name + " = {\n"
                f.write(head2.encode('ascii'))
                #write head3
                head3 = "//buffer data size: " + str(raw_data.size*4) + "\n"
                f.write(head3.encode('ascii'))

                for meta in raw_data.flat:
                    _ubyteArray = bytearray(struct.pack("f", meta))
                    f.write(_ubyteArray)

                i +=1
        f.close

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='caffe prototxt file')
    parser.add_argument('--weight_file', help='caffe caffemodel file')
    parser.add_argument('-i', '--image', help='input images')
    parser.add_argument('--dump_bin', default=False, type=str, help='Dump data save as binary (or string). True or False')
    parser.add_argument('-n', '--preprocess', type=str, default='0',
                    help='Name of model to select image preprocess method')
    args = parser.parse_args()
    model_name = args.preprocess
    image_path = args.image
    dump_bin = args.dump_bin
    norm = True

    if (dump_bin != "True"):
        file = './dumpData/NHWC/'
        mkdir(file)
        file = './dumpData/NCHW/'
        mkdir(file)
    else:
        file = './dumpData/'
        mkdir(file)

    preprocess_funcs = [image_preprocess_func(n) for n in model_name.split(',')]

    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        dir_name = None
        base_name = image_path
    if not os.path.exists(base_name):
        raise ValueError('No such {} image or directory.'.format(base_name))

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


    net = caffe.Net(args.model_file, args.weight_file, caffe.TEST)

    if len(preprocess_funcs) > 1:
        images = image_list[0]
        if len(image_list[0]) != len(preprocess_funcs):
            raise ValueError('Got different num of preprocess_methods and images!')
    else:
        if isinstance(image_list[0], list):
            images = image_list[0]
            if len(preprocess_funcs) != 1 or len(image_list[0]) != 1:
                raise ValueError('Got different num of preprocess_methods and images!')
        else:
            images = [image_list[0]]

    extract_feature(net, images, preprocess_funcs, dump_bin)

    print("Caffe dump data done. See results in ./dumpData")

