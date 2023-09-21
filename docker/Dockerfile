FROM ubuntu:18.04
ENV LESSCHARSET=utf-8
ENV LANG=C.UTF-8
COPY get-pip-27.py get-pip-37.py /tmp
RUN apt-get update && apt-get -y install python2.7 libpython2.7 python3.7 python3.7-distutils libglib2.0-dev libxrender-dev libsm6 libxext6 \
 && python2.7 /tmp/get-pip-27.py && rm -f /tmp/get-pip-27.py \
 && python3.7 /tmp/get-pip-37.py && rm -f /tmp/get-pip-37.py \
 && python2.7 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && python3.7 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip2 install torch==1.0.0 future==0.18.3 scikit-image==0.14.5 protobuf==3.3.0
RUN pip3 install numpy==1.17.5 \
 && pip3 install protobuf==3.19.6 \
 && pip3 install pyyaml==5.3.1 \
 && pip3 install enum34==1.1.10 \
 && pip3 install cython==0.29.36 \
 && pip3 install pycocotools==2.0.7 \
 && pip3 install matplotlib==3.3.4 \
 && pip3 install scipy==1.3.3 \
 && pip3 install joblib==1.0.1 \
 && pip3 install pulp==2.3.1 \
 && pip3 install opencv-python==4.2.0.34 \
 && pip3 install onnx==1.8.1 \
 && pip3 install onnxruntime==1.5.2 \
 && pip3 install onnx-simplifier==0.3.10 \
 && pip3 install onnxoptimizer==0.2.7 \
 && pip3 install torch==1.8.1 \
 && pip3 install torchvision==0.9.1 \
 && pip3 install tensorflow==1.15.5
