#!/bin/bash

if [ $# -lt 1 ]; then
	echo "$0 <dir>"
	echo ""
	exit -1
fi

source_dir=$1


top_dir=`pwd`
bin_file=${top_dir}/splite_binary_to_string

cd ${source_dir}
dir=`pwd`
filelist=`ls $dir`
for file in $filelist
do
  if [[ $file == *".c" ]]; then
		continue
	fi
	${bin_file} ${file}
done
echo "done"