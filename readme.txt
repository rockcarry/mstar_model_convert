
mstar 平台神经网络模型转换工具


+------------+
 环境搭建方法
+------------+

1. 需要 ubuntu 18.04 + python2 + python3 的环境（python3 的版本建议为 3.7.10）
建议使用 apical/aicnn-dev docker 镜像

2. 执行 install_requirements.sh 安装 python 库
./install_requirements.sh
（只需要执行一次即可，需要较长时间请耐心等待）

3. 执行 envsetup.sh 设置环境变量
source envsetup.sh
（登陆 shell 后只需要执行一次即可）


+-----------------+
 darknet 转 caffe
+-----------------+

使用 darknet2caffe 工具可以将 darknet 的 .cfg + .weights 模型转换为 caffe 的 .prototxt 和 .caffemodel 模型
（需要 ubuntu 18.04 + python2 的环境）

python2 $SGS_IPU_DIR/Scripts/darknet2caffe/darknet2caffe.py \
yolo-fastest-1.1-xl.cfg yolo-fastest-1.1-xl.weights \
yolo-fastest-1.1-xl.prototxt yolo-fastest-1.1-xl.caffemodel


+------------------+
 caffe 转 sgs float
+------------------+

使用 ConvertTool.py 工具，可以将 caffe 模型转换为 sgs 的 float 模型
（需要 ubuntu 18.04 + python3 的环境，建议使用我们的 docker 镜像）

python3 $SGS_IPU_DIR/Scripts/ConvertTool/ConvertTool.py caffe \
--model_file  $PWD/yolo-fastest-1.1-xl.prototxt   \
--weight_file $PWD/yolo-fastest-1.1-xl.caffemodel \
--input_arrays  data \
--output_arrays layer121-conv,layer130-conv \
--input_config $PWD/input_config.ini \
--output_file  $PWD/yolo-fastest-1.1-xl-float.sim

input_config.ini 配置文件如下：
[INPUT_CONFIG]
inputs          = data;
input_formats   = RGB;
quantizations   = TRUE;
mean_red        = 0;
mean_green      = 0;
mean_blue       = 0;
std_value       = 255;

[OUTPUT_CONFIG]
outputs         = layer121-conv,layer130-conv;
dequantizations = TRUE,TRUE;

[CONV_CONFIG]
input_format    = ALL_INT16;

注意事项：
1. 配置文件中 inputs  的值和命令行中 --input_arrays  的值要保持一致，建议去掉引号
2. 配置文件中 outputs 的值和命令行中 --output_arrays 的值要保持一致，建议去掉引号。有多个 output 要用逗号分隔
3. 配置文件中 outputs 如果有多个输出，dequantizations 也需要配置多个 TRUE 用逗号分隔
4. inputs 和 outputs 请根据具体的网络模型进行配置（使用 netron 工具查看节点名称）


+-------------------+
 sgs float 转 fixed
+-------------------+

使用 calibrator.py 工具，可以将 sgs float 模型转换为 sgs 的 fixed 模型
（需要 ubuntu 18.04 + python3 的环境，建议使用我们的 docker 镜像）

python3 $SGS_IPU_DIR/Scripts/calibrator/calibrator.py \
-i $SGS_IPU_DIR/images \
-m $PWD/yolo-fastest-1.1-xl-float.sim \
-o $PWD/yolo-fastest-1.1-xl-fixed.sim \
-c Unknown \
-n caffe_yolo_fastest \
--quant_level L5 \
--input_config $PWD/input_config.ini

注意事项：
1. 参数 -i $SGS_IPU_DIR/images 指定了用于量化计算的图片，在这个路径下需要存放一定数量的图片，用于做量化计算
2. input_config.ini 配置文件中，要正确配置 mean_red、mean_green、mean_blue 和 std_value 的值，否则量化会报错（参考原厂提供的 SGS_Models.bz2）


+---------------------+
 sgs fixed 转 offline
+---------------------+

使用 compiler.py 工具，可以将 sgs fixed 模型转换为 sgs offline 模型，可用于在 IPU 上部署运行
（需要 ubuntu 18.04 + python3 的环境，建议使用我们的 docker 镜像）

python3 $SGS_IPU_DIR/Scripts/calibrator/compiler.py \
-m $PWD/yolo-fastest-1.1-xl-fixed.sim   \
-o $PWD/yolo-fastest-1.1-xl-offline.sim \
-c Unknown


+--------+
 模拟验证
+--------+

使用 simulator.py 对模型进行模拟验证
（需要 ubuntu 18.04 + python3 的环境，建议使用我们的 docker 镜像）

python3 $SGS_IPU_DIR/Scripts/calibrator/simulator.py \
-i $PWD/test.jpg \
-m $PWD/yolo-fastest-1.1-xl-offline.sim \
-c Unknown \
-t Offline \
-n caffe_yolo_fastest


+------------------+
 postpc 后处理程序
+------------------+

postpc 是一个 c 语言编写的模型输出后处理程序，目前可以对 yolov3 yolo-fastest 模型进行目标矩形框的计算。

使用方法：
postpc log/output/unknown_yolo-fastest-1.1-xl-fixed.sim_test.bmp.txt 640 424

其中 unknown_yolo-fastest-1.1-xl-fixed.sim_test.bmp.txt 文件是使用 simulator.py 仿真运行后得到的模型输出文件，里面保存了NHWC 格式的 tensor 数据。
640 和 424 指定了 test.jpg 的图像宽度和高度，计算 BBOX 的时候需要这两个参数。


+--------------------------------+
 apical/aicnn-dev docker 使用说明
+--------------------------------+

docker run -it --rm -u $(id -u):$(id -g) \
-v /home:/home \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro   \
-v /etc/shadow:/etc/shadow:ro \
apical/aicnn-dev:v1.0.0 bash



chenk@apical.com.cn
9:04 2021/9/18







