#!/bin/bash

# Run script with multi-process. Set '0' to disable multi-process, '1' to enable.
ENABLE_MULTI_PROCESS=0
# Limit max process according to the actual situation. Default max process number is 4.
MAX_PROCESS_WORK_WITH=4

threshold_value=1.00

CUR_PATH=`pwd`
out_dir=""
BENCHMARK_LAYER_LIST=""
GEN_LAYER_LIST=""
top_dir=""
float_sim=""
fixed_sim=""
sample_bin=""
benchmark_bin=""
img_file=""
label_file=""
net_type="Undefined"
pre_process=""
compare_mode=""

function getTiming() {
    start=$1
    end=$2
    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)
# for debug..
#    echo $start
#    echo $end
    time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))
    echo "$time ms"
}

function printWarning() {
	echo "**********************************************************************************"
	echo "When 'auto_dump_debug.sh' runs on float vs fixed sim mode, SEVEN params mandatory:"
	echo "01. FULL-path of 'SGS_IPU_SDK', or put it here"
	echo "02. FULL-path of float sim, or put it here"
	echo "03. FULL-path of fixed sim, or put it here"
	echo "04. FULL-path of picture file, or put it here"
	echo "05. FULL-path of label file, or put it here"
	echo "06. Type of the net which should be 'Classification', 'Detection' or 'Unknown'"
	echo "07. FULL-path of the net pre-process python file, or put it here"
	echo "*****************************************"
	echo "Example 1 for giving a FULL-path:"
	echo "You could try command like:"
	echo "/bin/bash ./xxxx/auto_dump_debug.sh  /home/user/SGS_IPU_SDK  /home/user/float.sim \\"
	echo "/home/user/fixed.sim  /home/user/image.bmp  /home/user/label.txt  Classification  ./xxxx/SGS_Models/tensorflow/mobilenet_v1/mobilenet_v1.py"
	echo "*****************************************"
	echo "Example 2 for all param files are already here:"
	echo "/bin/bash SGS_IPU_SDK/DumpDebug/auto_dump_debug.sh  SGS_IPU_SDK  float.sim  fixed.sim \\"
	echo "image.bmp  label.txt  Classification  mobilenet_v1.py"
	echo "*****************************************"
	echo "When 'auto_dump_debug.sh' runs on sample vs benchmark dump bin mode, THREE params mandatory:"
	echo "01. FULL-path of 'SGS_IPU_SDK', or put it here"
	echo "02. FULL-path of dumped sample bin, or put it here"
	echo "03. FULL-path of dumped benchmark bin, or put it here"
	echo "*****************************************"
	echo "Example 3 for giving a FULL-path:"
	echo "You could try command like:"
	echo "/bin/bash ./xxxx/auto_dump_debug.sh  /home/user/SGS_IPU_SDK  /home/user/sample.bin  /home/user/benchmark.bin"
	echo "*****************************************"
	echo "Example 4 for all param files are already here:"
	echo "/bin/bash SGS_IPU_SDK/DumpDebug/auto_dump_debug.sh  SGS_IPU_SDK  sample.bin  benchmark.bin"
	echo "**********************************************************************************"
}

function sedBuild() {
	local out_dir=$1
	local out_list_item=$2
	local top_dir=$3

	cd ${out_dir}/genCompareLayer/${out_list_item};
	if [ ${ENABLE_MULTI_PROCESS} -eq 0 ]; then
		sed -i 's@\(.*\) = {@// \1\ndouble op_out_force_data2[] = {@g' ${out_dir}/genCompareLayer/${out_list_item}/*_benchmark.bin
	else
		sed -i 's@\(.*\) = {@// \1\ndouble op_out_force_data2[] = {@g' ${out_dir}/genCompareLayer/${out_list_item}/*_benchmark.bin &
	fi
	sed -i 's@\(.*\) = {@// \1\ndouble op_out_force_data[] = {@g' ${out_dir}/genCompareLayer/${out_list_item}/*_sample.bin
	wait
	echo "// ${out_dir}/genCompareLayer/${out_list_item}/*_sample.bin" > force_data.c
	echo "// ${out_dir}/genCompareLayer/${out_list_item}/*_benchmark.bin" > force_data2.c
	cat ${out_dir}/genCompareLayer/${out_list_item}/*_sample.bin >> force_data.c
	cat ${out_dir}/genCompareLayer/${out_list_item}/*_benchmark.bin >> force_data2.c

	#gcc ~/getData.c -I . -o getdata
	#gcc ~/getDiff.c -I . -o getdiff
	#gcc ~/getMax.c -I . -o getmax
	gcc ${top_dir}/DumpDebug/code/getRMse.c -I . -o getrmse -lm
}

function doAnalyzeRmse() {
	local firstEnter=1
	local lastRmse=0
	local ret=0.0
	local ret_grep
	local delAfter
	local delBefore
	local transformStr
	if [ -f ${GEN_LAYER_LIST} ]; then
		mkdir -vp ${out_dir}/doAnalyzeRmse
		source ${GEN_LAYER_LIST}
		#for out_list_item in ${out_list}
		#do
		#	if [ ${ENABLE_MULTI_PROCESS} -eq 0 ]; then
		#		sedBuild ${out_dir} ${out_list_item} ${top_dir}
		#	else
		#		while [ `ps -ef|grep sed|wc -l` -gt ${MAX_PROCESS_WORK_WITH} ]
		#		do
		#			echo "Task > ${MAX_PROCESS_WORK_WITH} sleep 1" && sleep 1
		#		done

		#		memInfo=`cat /proc/meminfo | grep MemAvailable | awk {'print $2'}`
		#		while [ `echo "${memInfo} < 1048576"|bc` -eq 1 ]
		#		do
		#			echo "${memInfo} < 1048576 sleep 1, low memory treat as fail" && sleep 1 >> ${out_dir}/${SHORT_REPORT}
		#			memInfo=`cat /proc/meminfo | grep MemAvailable | awk {'print $2'}`
		#		done

		#		sedBuild ${out_dir} ${out_list_item} ${top_dir} &
		#	fi
		#done
		#wait
		cd ${out_dir}/genCompareLayer/;
		cur_dir=`pwd`
		echo "${cur_dir}" > ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log
		for out_list_item in ${out_list}
		do
			cd ${out_dir}/genCompareLayer/${out_list_item};
			cur_dir=`pwd`
			echo -n -e "${cur_dir##*/}\t" > ${out_dir}/doAnalyzeRmse/rmseStringCur.log
			#ret_grep=`grep -i "op_out\[" ./sample.bin`
			#delAfter=${ret_grep#*']'}
			#delBefore=${delAfter%=*}
			#transformStr=${delBefore//\//_}
			#echo -n -e "OP_TYPE: ${transformStr}\t" >> ${out_dir}/doAnalyzeRmse/rmseStringCur.log
            #${top_dir}/DumpDebug/getrmse  sample.bin benchmark.bin >> ${out_dir}/doAnalyzeRmse/rmseStringCur.log
			#ret=`grep -i "RMSE:" ${out_dir}/doAnalyzeRmse/rmseStringCur.log`
			${top_dir}/DumpDebug/compareAccuracy  sample.bin benchmark.bin >> ${out_dir}/doAnalyzeRmse/rmseStringCur.log
			ret=`grep -i "OP: " ${out_dir}/doAnalyzeRmse/rmseStringCur.log`
			if [ "${ret}" ]; then
				delBefore=${ret#*RMSE:	}
				if [ 1 -eq ${firstEnter} ]; then
					lastRmse=${delBefore}
					mv ${out_dir}/doAnalyzeRmse/rmseStringCur.log ${out_dir}/doAnalyzeRmse/rmseStringLast.log
					firstEnter=0
				else
					ret=$(echo "${delBefore} - ${lastRmse}"|bc)
					if [ `echo "${ret} > ${threshold_value}"|bc` -eq 1 ]; then
						echo -e "`head -c-1 ${out_dir}/doAnalyzeRmse/rmseStringCur.log`" "    **** OVER threshold!!!! ****" > ${out_dir}/doAnalyzeRmse/rmseStringCur_.log
						echo -e "`head -c-1 ${out_dir}/doAnalyzeRmse/rmseStringLast.log`" > ${out_dir}/doAnalyzeRmse/rmseStringLast_.log
						mv ${out_dir}/doAnalyzeRmse/rmseStringCur_.log ${out_dir}/doAnalyzeRmse/rmseStringCur.log
						mv ${out_dir}/doAnalyzeRmse/rmseStringLast_.log ${out_dir}/doAnalyzeRmse/rmseStringLast.log
					fi
					lastRmse=${delBefore}
					cat ${out_dir}/doAnalyzeRmse/rmseStringLast.log >> ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log
					mv ${out_dir}/doAnalyzeRmse/rmseStringCur.log ${out_dir}/doAnalyzeRmse/rmseStringLast.log
					echo > ${out_dir}/doAnalyzeRmse/rmseStringCur.log
				fi
			fi
			#rm ${out_dir}/genCompareLayer/${out_list_item}/*
		done
		if [ -f ${out_dir}/doAnalyzeRmse/rmseStringLast.log ]; then
			cat ${out_dir}/doAnalyzeRmse/rmseStringLast.log >> ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log
		fi
		if [ -f ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log ]; then
			cat ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log
			if [ ${compare_mode} == "sim_file" ]; then
				cp ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log ${out_dir}/${fixed_sim##*/}_doAnalyzeRmse_run.log
			else
				cp ${out_dir}/doAnalyzeRmse/doAnalyzeRmse_run.log ${out_dir}/${benchmark_bin##*/}_doAnalyzeRmse_run.log
			fi
		else
			echo "**********************************************************************************"
			echo "ERROR: doAnalyzeRmse fail or no output tensor match between two!!!"
			echo "Please check ${GEN_LAYER_LIST}"
			printWarning
		fi
	else
		echo "**********************************************************************************"
		echo "ERROR: ${GEN_LAYER_LIST} is not exist!!!"
		printWarning
	fi
}

function matchTensor() {
	local benchmark_layer_item=$1
	local ret
	local delAfter
	local delBefore
	local transformStr
	local transformStr_

	if [ ${compare_mode} == "sim_file" ]; then
		cd ${out_dir}/dumpFixed_out/output/${benchmark_layer_item}
		ret=`grep -i " name: " ./sigma_outtensor_dump_${fixed_sim##*/}_benchmark.bin`
		delAfter=${ret#*' name: '}
		delBefore=${delAfter%%' bConstant'*}
		transformStr_=${delBefore// /_}
		transformStr=${transformStr_//\//_}
		mv sigma_outtensor_dump_${fixed_sim##*/}_benchmark.bin ${transformStr}_${fixed_sim##*/}_benchmark.bin
		ret=`grep -lsr " name: ${delBefore} " ${out_dir}/dumpFloat_out/output/*.xx.output*/*`
		if [ -f ${ret} ] && [ "${ret}" ]; then
			cp -rf ${out_dir}/dumpFixed_out/output/${benchmark_layer_item} ${out_dir}/genCompareLayer
			mv ${ret} ${out_dir}/genCompareLayer/${benchmark_layer_item}/${transformStr}_${float_sim##*/}_sample.bin
		fi
	else
		cd ${out_dir}/benchmark_out/${benchmark_layer_item}
		ret=`grep -i " name: " ./${benchmark_bin##*/}_benchmark.bin`
		delAfter=${ret#*' name: '}
		delBefore=${delAfter%%' bConstant'*}
		transformStr_=${delBefore// /_}
		transformStr=${transformStr_//\//_}
		mv ${benchmark_bin##*/}_benchmark.bin ${transformStr}_${benchmark_bin##*/}_benchmark.bin
		ret=`grep -lsr " name: ${delBefore} " ${out_dir}/sample_out/*.xx.output*/*`
		if [ -f ${ret} ] && [ "${ret}" ]; then
			mv ${out_dir}/benchmark_out/${benchmark_layer_item} ${out_dir}/genCompareLayer
			mv ${ret} ${out_dir}/genCompareLayer/${benchmark_layer_item}/${transformStr}_${sample_bin##*/}_sample.bin
		fi
	fi
}

function pickFile() {
  local benchmark_layer_item=$1
  tmp=${benchmark_layer_item#*.}
  folder_name="*.${tmp}"
  ret=`find ${out_dir}/sample_out/ -type d -name "${folder_name}"`
  if [ -d ${ret} ] && [ "${ret}" ]; then
    cd ${out_dir}/benchmark_out/${benchmark_layer_item}
    mv * benchmark.bin
    cd ${ret}
    mv * sample.bin
    mv ${out_dir}/benchmark_out/${benchmark_layer_item} ${out_dir}/genCompareLayer
    mv ${ret}/* ${out_dir}/genCompareLayer/${benchmark_layer_item}/
    rm ${ret} -rf
  fi
  }
function genCompareLayer() {
	local ret
	local delAfter
	local delBefore
	local transformStr

	GEN_LAYER_LIST=${out_dir}/gen_layer_list.txt
	if [ -f ${BENCHMARK_LAYER_LIST} ]; then
		mkdir -vp ${out_dir}/genCompareLayer
		source ${BENCHMARK_LAYER_LIST}
		for benchmark_layer_item in ${benchmark_layer_list}
		do
			if [ ${ENABLE_MULTI_PROCESS} -eq 0 ]; then
				pickFile ${benchmark_layer_item}
			else
				while [ `ps -ef|grep lsr|wc -l` -gt ${MAX_PROCESS_WORK_WITH} ]
				do
					echo "Task > ${MAX_PROCESS_WORK_WITH} sleep 1" && sleep 1
				done

				memInfo=`cat /proc/meminfo | grep MemAvailable | awk {'print $2'}`
				while [ `echo "${memInfo} < 1048576"|bc` -eq 1 ]
				do
					echo "${memInfo} < 1048576 sleep 1, low memory treat as fail" && sleep 1 >> ${out_dir}/${SHORT_REPORT}
					memInfo=`cat /proc/meminfo | grep MemAvailable | awk {'print $2'}`
				done
				pickFile ${benchmark_layer_item} &
			fi
		done
		wait
		echo "out_list=\"" > ${GEN_LAYER_LIST}
		cd ${out_dir}/genCompareLayer; ls *.xx.output* -d -1 -v 2>/dev/null >> ${GEN_LAYER_LIST}
		echo "\"" >> ${GEN_LAYER_LIST}
	else
		echo "**********************************************************************************"
		echo "ERROR: ${BENCHMARK_LAYER_LIST} is not exist!!!"
		printWarning
	fi
}

function dumpFixed() {
	BENCHMARK_LAYER_LIST=${out_dir}/benchmark_layer_list.txt
	if [ ${compare_mode} == "sim_file" ]; then
		if [ -f ${top_dir}/Scripts/calibrator/simulator.py ]; then
			mkdir -vp ${out_dir}/benchmark_out; cd ${out_dir}/benchmark_out;
			echo "dumpTensor" > DebugConfig.txt; echo "eliminateGarbage" >> DebugConfig.txt;
			echo "dequantFixed" >> DebugConfig.txt; echo "path=." >> DebugConfig.txt;

			echo "python3 ${top_dir}/Scripts/calibrator/simulator.py -i ${img_file} -m ${fixed_sim} \
-l ${label_file} -c ${net_type} -n ${pre_process} -t Fixed 2>&1 > ${out_dir}/benchmark_out/dumpFixed_run.log"

			python3 ${top_dir}/Scripts/calibrator/simulator.py -i ${img_file} -m ${fixed_sim} \
			-l ${label_file} -c ${net_type} -n ${pre_process} -t Fixed 2>&1 > ${out_dir}/benchmark_out/dumpFixed_run.log

			mv sigma_outtensor_dump.bin sigma_outtensor_dump_${fixed_sim##*/}_benchmark.bin
      #cp ${top_dir}/DumpDebug/splite_binary_to_binary ./;
			${top_dir}/DumpDebug/splite_binary_to_binary sigma_outtensor_dump_${fixed_sim##*/}_benchmark.bin
		else
			echo "**********************************************************************************"
			echo "FATAL: dumpFixed() simulator.py not found!!!"
			printWarning
			return
		fi
	else
		mkdir -vp ${out_dir}/benchmark_out;
		cd ${out_dir}/benchmark_out; cp ${benchmark_bin} ${out_dir}/benchmark_out/${benchmark_bin##*/}_benchmark.bin -rf
		cp ${top_dir}/DumpDebug/splite_binary_to_binary ./;

		echo "${out_dir}/benchmark_out/splite_binary_to_binary ${benchmark_bin##*/}_benchmark.bin \
2>&1 > ${out_dir}/benchmark_out/benchmark_run.log"

		${out_dir}/benchmark_out/splite_binary_to_binary ${benchmark_bin##*/}_benchmark.bin 2>&1 > ${out_dir}/benchmark_out/benchmark_run.log
		rm ${benchmark_bin##*/}_benchmark.bin
	fi

	rm -rf *.xx.inputs;
	echo "benchmark_layer_list=\"" > ${BENCHMARK_LAYER_LIST}
	ls *.xx.output* -d -1 -v >> ${BENCHMARK_LAYER_LIST}
	echo "\"" >> ${BENCHMARK_LAYER_LIST}
}

function dumpFloat() {
	if [ ${compare_mode} == "sim_file" ]; then
		if [ -f ${top_dir}/Scripts/calibrator/simulator.py ]; then
			mkdir -vp ${out_dir}/sample_out;
			cd ${out_dir}/sample_out
			echo "dumpTensor" > DebugConfig.txt; echo "eliminateGarbage" >> DebugConfig.txt;
			echo "path=." >> DebugConfig.txt;

			echo "python3 ${top_dir}/Scripts/calibrator/simulator.py -i ${img_file} -m ${float_sim} \
-l ${label_file} -c ${net_type} -n ${pre_process} -t Float 2>&1 > ${out_dir}/sample_out/dumpFloat_run.log"

			python3 ${top_dir}/Scripts/calibrator/simulator.py -i ${img_file} -m ${float_sim} \
			-l ${label_file} -c ${net_type} -n ${pre_process} -t Float 2>&1 > ${out_dir}/sample_out/dumpFloat_run.log

			mv sigma_outtensor_dump.bin sigma_outtensor_dump_${float_sim##*/}_sample.bin
			#cp ${top_dir}/DumpDebug/splite_binary_to_binary ./
			${top_dir}/DumpDebug/splite_binary_to_binary sigma_outtensor_dump_${float_sim##*/}_sample.bin
		else
			echo "**********************************************************************************"
			echo "FATAL: dumpFloat() simulator.py not found!!!"
			printWarning
			return
		fi
	else
		mkdir -vp ${out_dir}/sample_out;
		cd ${out_dir}/sample_out; cp ${sample_bin} ${out_dir}/sample_out/${sample_bin##*/}_sample.bin -rf
		cp ${top_dir}/DumpDebug/splite_binary_to_binary ./;

		echo "${out_dir}/sample_out/splite_binary_to_binary ${sample_bin##*/}_sample.bin \
2>&1 > ${out_dir}/sample_out/sample_run.log"

		${out_dir}/sample_out/splite_binary_to_binary ${sample_bin##*/}_sample.bin 2>&1 > ${out_dir}/sample_out/sample_run.log
		rm ${sample_bin##*/}_sample.bin
	fi

	rm -rf *.xx.inputs;
}

function checkParam() {
	local ret=0

	echo "Param1: $1"
	echo "Param2: $2"
	echo "Param3: $3"
	echo "Param4: $4"
	echo "Param5: $5"
	echo "Param6: $6"
	echo "Param7: $7"

	if [ -f ${CUR_PATH}/$1/cfg_env.sh ]; then
		top_dir=${CUR_PATH}/$1
	elif [ -f $1/cfg_env.sh ]; then
		top_dir=$1
	else
		echo "**********************************************************************************"
		echo "FATAL: SGS_IPU_SDK not found!!!"
		printWarning
		ret=-1
	fi

	if [ 0 -eq ${ret} ]; then
		if [ -f ${CUR_PATH}/$2 ]; then
			if [[ $2 == *".sim" ]]; then
				float_sim=${CUR_PATH}/$2
				compare_mode="sim_file"
			elif [[ $2 == *".bin" ]]; then
				sample_bin=${CUR_PATH}/$2
				compare_mode="bin_file"
			fi
		elif [ -f $2 ] && [ "$2" ]; then
			if [[ $2 == *".sim" ]]; then
				float_sim=$2
				compare_mode="sim_file"
			elif [[ $2 == *".bin" ]]; then
				sample_bin=$2
				compare_mode="bin_file"
			fi
		fi
		if [[ -z "${compare_mode}" ]]; then
			echo "**********************************************************************************"
			echo "FATAL: float sim or sample dump bin file not exist!!!"
			printWarning
			ret=-1
		fi
	fi

	if [ 0 -eq ${ret} ]; then
		if [ -f ${CUR_PATH}/$3 ]; then
			if [ ${compare_mode} == "sim_file" ] && [[ $3 == *".sim" ]]; then
				fixed_sim=${CUR_PATH}/$3
			elif [ ${compare_mode} == "bin_file" ] && [[ $3 == *".bin" ]]; then
				benchmark_bin=${CUR_PATH}/$3
			fi
		elif [ -f $3 ] && [ "$3" ]; then
			if [ ${compare_mode} == "sim_file" ] && [[ $3 == *".sim" ]]; then
				fixed_sim=$3
			elif [ ${compare_mode} == "bin_file" ] && [[ $3 == *".bin" ]]; then
				benchmark_bin=$3
			fi
		fi
		if [[ -z "${fixed_sim}" ]] && [[ -z "${benchmark_bin}" ]]; then
			echo "**********************************************************************************"
			echo "FATAL: fixed sim or benchmark dump bin file not exist!!!"
			printWarning
			ret=-1
		fi
	fi

	if [ 0 -eq ${ret} ] && [ "${compare_mode}" ] && [ ${compare_mode} == "sim_file" ]; then
		if [ -f ${CUR_PATH}/$4 ]; then
			img_file=${CUR_PATH}/$4
		elif [ -f $4 ] && [ "$4" ]; then
			img_file=$4
		else
			echo "**********************************************************************************"
			echo "FATAL: picture file not exist!!!"
			printWarning
			ret=-1
		fi
	fi

	if [ 0 -eq ${ret} ] && [ "${compare_mode}" ] && [ ${compare_mode} == "sim_file" ]; then
		if [ -f ${CUR_PATH}/$5 ]; then
			label_file=${CUR_PATH}/$5
		elif [ -f $5 ] && [ "$5" ]; then
			label_file=$5
		else
			echo "**********************************************************************************"
			echo "FATAL: label file not exist!!!"
			printWarning
			ret=-1
		fi
	fi

	if [ 0 -eq ${ret} ] && [ "${compare_mode}" ] && [ ${compare_mode} == "sim_file" ]; then
		if [ "$6" ]; then
			net_type=$6
		fi
		if [ "Classification" != ${net_type} ] && [ "Detection" != ${net_type} ] && [ "Unknown" != ${net_type} ]; then
			echo "**********************************************************************************"
			echo "FATAL: Net type of ${fixed_sim} unknown!!!"
			printWarning
			ret=-1
		fi
	fi

	if [ 0 -eq ${ret} ] && [ "${compare_mode}" ] && [ ${compare_mode} == "sim_file" ]; then
		if [ "$7" ]; then
			pre_process=$7
		else
			echo "**********************************************************************************"
			echo "FATAL: Pre-process of picture for ${fixed_sim} unknown!!!"
			printWarning
			ret=-1
		fi
	fi

	return ${ret}
}

function dumpDebug() {
	local ret

	checkParam $1 $2 $3 $4 $5 $6 $7
	ret=$?
	if [ 0 -eq ${ret} ]; then
		if [ ${compare_mode} == "sim_file" ]; then
			out_dir=${CUR_PATH}/${net_type}_${fixed_sim##*/}_DumpDebug_out
		else
			out_dir=${CUR_PATH}/${net_type}_${benchmark_bin##*/}_DumpDebug_out
		fi
		if [ -d ${out_dir} ]; then
			mv ${out_dir} ${out_dir}_backup_at_`date +%Y-%m-%d_%H-%M-%S`; mkdir -vp ${out_dir};
		else
			mkdir -vp ${out_dir};
		fi
		cd ${top_dir}; source cfg_env.sh;
		# label image tf
		if [ ${ENABLE_MULTI_PROCESS} -eq 0 ]; then
			start=$(date +%s.%N)
			dumpFloat
			end=$(date +%s.%N)
			echo 'split FLOAT #####################'
			getTiming $start $end
		else
			dumpFloat &
		fi
		# label image fix
		start=$(date +%s.%N)
		dumpFixed
		end=$(date +%s.%N)
		echo 'split FIXED #####################'
		getTiming $start $end
		wait
		#exit
		# compare
		start1=$(date +%s.%N)
		genCompareLayer
		end1=$(date +%s.%N)
		echo 'genCompareLayer #####################'
		getTiming $start1 $end1

		# sh
		start2=$(date +%s.%N)
		doAnalyzeRmse
		end2=$(date +%s.%N)
		echo 'doAnalyzeRmse #####################'
		getTiming $start2 $end2
		echo ""
		echo "**********************************************************************************"
		echo "DumpDebug done!!!"
	fi
}

dumpDebug $1 $2 $3 $4 $5 $6 $7
