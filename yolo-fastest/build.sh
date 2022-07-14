#!/bin/bash

set -e

echo "convert yolo-fastest from darknet to caffe..."
python2 $SGS_IPU_DIR/Scripts/darknet2caffe/darknet2caffe.py \
yolo-fastest-1.1-xl.cfg yolo-fastest-1.1-xl.weights \
yolo-fastest-1.1-xl.prototxt yolo-fastest-1.1-xl.caffemodel

echo "convert yolo-fastest from caffe to sgs float ..."
python3 $SGS_IPU_DIR/Scripts/ConvertTool/ConvertTool.py caffe \
--model_file  $PWD/yolo-fastest-1.1-xl.prototxt   \
--weight_file $PWD/yolo-fastest-1.1-xl.caffemodel \
--input_arrays  data \
--output_arrays layer121-conv,layer130-conv \
--input_config $PWD/input_config.ini \
--output_file  $PWD/yolo-fastest-1.1-xl-float.sim

echo "convert yolo-fastest from sgs float to sgs fixed ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/calibrator.py \
-i $SGS_IPU_DIR/images \
-m $PWD/yolo-fastest-1.1-xl-float.sim \
-o $PWD/yolo-fastest-1.1-xl-fixed.sim \
-c Unknown \
-n caffe_yolo_fastest \
--quant_level L5 \
--input_config $PWD/input_config.ini

echo "convert yolo-fastest from sgs fixed to sgs offline ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/compiler.py \
-m $PWD/yolo-fastest-1.1-xl-fixed.sim   \
-o $PWD/yolo-fastest-1.1-xl-offline.sim \
-c Unknown

echo "simulator run sgs offline model ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/simulator.py \
-i $PWD/test.jpg \
-m $PWD/yolo-fastest-1.1-xl-offline.sim \
-c Unknown \
-t Offline \
-n caffe_yolo_fastest

echo "post process simulator output log, and get detection result ..."
postpc $PWD/log/output/unknown_yolo-fastest-1.1-xl-offline.sim_test.jpg.txt 640 424
