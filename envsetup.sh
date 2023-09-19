#!/bin/bash

avx2_support=`cat /proc/cpuinfo | grep ' avx2 ' | wc -l`

if [ $avx2_support -eq "0" ];then
   echo "error not support avx2 !!!!, please choose Intel cpu which support avx2 "
fi

SGS_IPU_DIR=`pwd`

if [ ! -d "${SGS_IPU_DIR}/libs/x86_64" ]; then
    echo "error! please enter the SGS Release Dir !"
else
    export SGS_IPU_DIR
    export PATH=$PATH:$SGS_IPU_DIR/postpc
    export LD_LIBRARY_PATH=${SGS_IPU_DIR}/libs/x86_32:${SGS_IPU_DIR}/libs/x86_64:${LD_LIBRARY_PATH}
    export PYTHONPATH=${SGS_IPU_DIR}/Scripts:${SGS_IPU_DIR}/Scripts/darknet2caffe/pycaffe:${PYTHONPATH}
fi

