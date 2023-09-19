#!/bin/bash

if [ $# -lt 2 ]; then
    echo "$0 <exec_file path> <core_file path>"
    exit -1
fi

EXEC_PATH=$1
CORE_PATH=$2

gdb -ex "info proc mappings" --batch ${EXEC_PATH} ${CORE_PATH}
gdb -ex bt --batch ${EXEC_PATH} ${CORE_PATH}
