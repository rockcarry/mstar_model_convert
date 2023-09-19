#!/bin/bash

source_a=sigma_outtensor_dump.bin
source_b=float32.run.dump

top_dir=`pwd`
bin_file=${top_dir}/splite_dump
sgs_net_file1=/home/jack.deng/jack.deng/test/cmodel_fixed/resnet/12227_16_v2_300.tflite
sgs_net_file2=/home/jack.deng/jack.deng/test/cmodel_fixed/resnet/12227_16_v2.tflite
tflite_float_net_file=/home/jack.deng/jack.deng/test/cmodel_float/resnet_v2/12227_float.tflite
tflite_float_net_file1=/home/jack.deng/jack.deng/test/cmodel_fixed/resnet/int8/12227_fix8.tflite
label_path=/home/jack.deng/jack.deng/test/label.txt
toco_bin=/home/jack.deng/jack.deng/src/tensorflow-tip/tensorflow/bazel-bin/tensorflow/lite/examples/label_image/label_image
sgs_bin=./label_image

export LD_LIBRARY_PATH=`pwd`:${LD_LIBRARY_PATH}

test_error_set_list=`find ${top_dir} -name "*.bmp"`
out_path=${top_dir}/out/
mkdir -vp ${out_path}



out_list="
0.PAD.xx
1.CONV_2D.xx
2.MAX_POOL_2D.xx
3.MUL.xx
4.ADD.xx
5.CONV_2D.xx
6.CONV_2D.xx
7.CONV_2D.xx
9.ADD.xx
10.MUL.xx
11.ADD.xx
12.CONV_2D.xx
13.CONV_2D.xx
14.CONV_2D.xx
15.ADD.xx
16.MUL.xx
17.ADD.xx
18.MAX_POOL_2D.xx
19.CONV_2D.xx
20.PAD.xx
21.CONV_2D.xx
22.CONV_2D.xx
23.ADD.xx
24.MUL.xx
25.ADD.xx
26.CONV_2D.xx
27.CONV_2D.xx
28.CONV_2D.xx
29.CONV_2D.xx
30.ADD.xx
31.MUL.xx
32.ADD.xx
33.CONV_2D.xx
34.CONV_2D.xx
35.CONV_2D.xx
36.ADD.xx
37.MUL.xx
38.ADD.xx
39.CONV_2D.xx
40.CONV_2D.xx
41.CONV_2D.xx
42.ADD.xx
43.MUL.xx
44.ADD.xx
45.MAX_POOL_2D.xx
46.CONV_2D.xx
47.PAD.xx
48.CONV_2D.xx
49.CONV_2D.xx
50.ADD.xx
51.MUL.xx
52.ADD.xx
53.CONV_2D.xx
54.CONV_2D.xx
55.CONV_2D.xx
56.CONV_2D.xx
57.ADD.xx
58.MUL.xx
59.ADD.xx
60.CONV_2D.xx
61.CONV_2D.xx
62.CONV_2D.xx
63.ADD.xx
64.MUL.xx
65.ADD.xx
66.CONV_2D.xx
67.CONV_2D.xx
68.CONV_2D.xx
69.ADD.xx
70.MUL.xx
71.ADD.xx
72.CONV_2D.xx
73.CONV_2D.xx
74.CONV_2D.xx
75.ADD.xx
76.MUL.xx
77.ADD.xx
78.CONV_2D.xx
79.CONV_2D.xx
80.CONV_2D.xx
81.ADD.xx
82.MUL.xx
83.ADD.xx
84.MAX_POOL_2D.xx
85.CONV_2D.xx
86.PAD.xx
87.CONV_2D.xx
88.CONV_2D.xx
89.ADD.xx
8.CONV_2D.xx
90.MUL.xx
91.ADD.xx
92.CONV_2D.xx
93.CONV_2D.xx
94.CONV_2D.xx
95.CONV_2D.xx
96.ADD.xx
97.MUL.xx
98.ADD.xx
99.CONV_2D.xx
100.CONV_2D.xx
101.CONV_2D.xx
102.ADD.xx
103.MUL.xx
104.ADD.xx
105.CONV_2D.xx
106.CONV_2D.xx
107.CONV_2D.xx
108.ADD.xx
109.MUL.xx
110.ADD.xx
111.MEAN.xx
112.CONV_2D.xx
113.RESHAPE.xx
114.SOFTMAX.xx
"
function func_getrmse()
{
	data_dir=$1
	cd ${data_dir}

	${bin_file} ${source_a}
	${bin_file} ${source_b}

	for fn in ${out_list}
	do
		cd ${data_dir}/$fn;
		sed -i 's@\(.*\) = {@// \1\ndouble op_out_force_data[] = {@g' ${source_a}
		sed -i 's@\(.*\) = {@// \1\ndouble op_out_force_data2[] = {@g' ${source_b}
		echo "// ${source_a}" > force_data.c
		echo "// ${source_b}" > force_data2.c
		cat ${source_b} >> force_data2.c
		cat ${source_a} >> force_data.c
	
		rm ${source_a}
		rm ${source_b}

		#gcc ~/getData.c -I . -o getdata
		#gcc ~/getDiff.c -I . -o getdiff
		#gcc ~/getMax.c -I . -o getmax
		gcc ${top_dir}/code/getRMse.c -I . -o getrmse
		cur_dir=`pwd`
		echo -n -e "${cur_dir}\t"
		./getrmse
	done
}

function eval_diff()
{
	local test_case=""
	local sgs_net_file=$1
	local out_dir=$2/`basename ${sgs_net_file}`
	local count=0
	for fn in ${test_error_set_list}
	do
		if [ ${count} -gt 5 ];then
			return
		fi
		((count=count+1))
		echo "Run: pic $count"
		test_case=${out_dir}/`basename ${fn}`
		mkdir -vp ${test_case}

		echo "Run toco evalute"
		echo ${toco_bin} -i $fn -m ${tflite_float_net_file} -l ${label_path}
		time ${toco_bin} -i $fn -m ${tflite_float_net_file} -l ${label_path} 2>&1 | tee ${test_case}/toco.log
		mv ~/float32.run.dump ${test_case}

		echo "Run sgs_interpreter evalute"
		echo ${sgs_bin} ${sgs_net_file} $fn ${label_path}
		time ${sgs_bin} ${sgs_net_file} $fn ${label_path} 2>&1 | tee ${test_case}/sgs_run.log
		mv ~/sigma_outtensor_dump.bin ${test_case}

		time func_getrmse ${test_case} 2>&1 | tee ${test_case}/getrmse.log
	done
}

function run_rmse()
{
	out_dir=$1
	for fn in ${out_list}
	do
		cd ${out_dir}/$fn
		cur_dir=`pwd`
		echo -n -e "${cur_dir}\t"
		./getrmse
	done

}
#time eval_diff ${sgs_net_file1} ${out_path}
time eval_diff ${sgs_net_file2} ${out_path}
#time eval_diff ${sgs_net_file3} ${out_path}
