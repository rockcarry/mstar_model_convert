#!/bin/bash

TYPE=xl
CLASS_NUM=80

if [ "$1"x == "body"x ]; then
    TYPE=body
    CLASS_NUM=1
fi

set -e

echo "convert yolo-fastest from darknet to caffe..."
python2 $SGS_IPU_DIR/Scripts/darknet2caffe/darknet2caffe.py \
yolo-fastest-1.1-$TYPE.cfg yolo-fastest-1.1-$TYPE.weights \
yolo-fastest-1.1-$TYPE.prototxt yolo-fastest-1.1-$TYPE.caffemodel

echo "convert yolo-fastest from caffe to sgs float ..."
python3 $SGS_IPU_DIR/Scripts/ConvertTool/ConvertTool.py caffe \
--model_file  $PWD/yolo-fastest-1.1-$TYPE.prototxt   \
--weight_file $PWD/yolo-fastest-1.1-$TYPE.caffemodel \
--input_arrays  data \
--output_arrays layer121-conv,layer130-conv \
--input_config $PWD/input_config.ini \
--output_file  $PWD/yolo-fastest-1.1-$TYPE-float.sim

echo "convert yolo-fastest from sgs float to sgs fixed ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/calibrator.py \
-i $SGS_IPU_DIR/images \
-m $PWD/yolo-fastest-1.1-$TYPE-float.sim \
-o $PWD/yolo-fastest-1.1-$TYPE-fixed.sim \
-c Unknown \
-n caffe_yolo_fastest \
--quant_level L3 \
--input_config $PWD/input_config.ini

echo "convert yolo-fastest from sgs fixed to sgs offline ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/compiler.py \
-m $PWD/yolo-fastest-1.1-$TYPE-fixed.sim   \
-o $PWD/yolo-fastest-1.1-$TYPE-offline.sim \
-c Unknown

echo "simulator run sgs offline model ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/simulator.py \
-i $PWD/test.jpg \
-m $PWD/yolo-fastest-1.1-$TYPE-offline.sim \
-c Unknown \
-t Offline \
-n caffe_yolo_fastest

echo "post process simulator output log, and get detection result ..."
postpc $PWD/log/output/unknown_yolo-fastest-1.1-$TYPE-offline.sim_test.jpg.txt 640 424 yolo-fastest ${CLASS_NUM}
