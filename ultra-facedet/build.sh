#!/bin/bash

set -e

echo "convert ultra-facedet from caffe to sgs float ..."
python3 $SGS_IPU_DIR/Scripts/ConvertTool/ConvertTool.py caffe \
--model_file  $PWD/RFB-320.prototxt   \
--weight_file $PWD/RFB-320.caffemodel \
--input_arrays  input \
--output_arrays scores,boxes \
--input_config $PWD/input_config.ini \
--output_file  $PWD/RFB-320-float.sim

echo "convert ultra-facedet from sgs float to sgs fixed ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/calibrator.py \
-i $SGS_IPU_DIR/images \
-m $PWD/RFB-320-float.sim \
-o $PWD/RFB-320-fixed.sim \
-c Unknown \
-n caffe_yolo_fastest \
--quant_level L3 \
--input_config $PWD/input_config.ini

echo "convert ultra-facedet from sgs fixed to sgs offline ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/compiler.py \
-m $PWD/RFB-320-fixed.sim   \
-o $PWD/RFB-320-offline.sim \
-c Unknown

echo "simulator run sgs offline model ..."
python3 $SGS_IPU_DIR/Scripts/calibrator/simulator.py \
-i $PWD/test.jpg \
-m $PWD/RFB-320-offline.sim \
-c Unknown \
-t Offline \
-n caffe_ultra_fast_face

postpc $PWD/log/output/unknown_RFB-320-offline.sim_test.jpg.txt 806 484 ultra-facedet
