# Copyright 2018 The MACE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import os
import sys
import calibrator_custom
import pkg_resources as pkg

AUTOGEN_INI_MODIFY_COUNT = 0

def mace_check(condition, msg):
    if not condition:
        raise Exception(msg)

def getIPUVersion():
    Version = calibrator_custom.utils.get_sdk_version()
    IPUVersion = ''
    if Version == '1':
      IPUVersion = 'I6E'    # IPU100
    elif Version == 'Q_0':
      IPUVersion = 'M6'     # IPU200M
    elif Version == 'S6':
      IPUVersion = 'I7'     # IPU260
    elif Version == 'S2':
      IPUVersion = 'M6P'    # IPU220
    elif Version == 'S1':
      IPUVersion = 'I6C'    # IPU201
    elif Version == 'S3':
      IPUVersion = 'P5'     # IPU230
    elif Version == 'S31':
      IPUVersion = 'I6F'    # IPU231
    elif Version == 'L10':
      IPUVersion = 'I6DC'   # IPU410
    elif Version == 'S02':
      IPUVersion = 'I6D'    # IPU202
    else:
      print('Not Found version: {}'.format(calibrator_custom.__version__))
    return IPUVersion


def roundup_div4(value):
    return int((value + 3) // 4)


class OpenCLBufferType(enum.Enum):
    CONV2D_FILTER = 0
    IN_OUT_CHANNEL = 1
    ARGUMENT = 2
    IN_OUT_HEIGHT = 3
    IN_OUT_WIDTH = 4
    WINOGRAD_FILTER = 5
    DW_CONV2D_FILTER = 6
    WEIGHT_HEIGHT = 7
    WEIGHT_WIDTH = 8


def calculate_image_shape(buffer_type, shape, winograd_blk_size=0):
    # keep the same with mace/kernel/opencl/helper.cc
    image_shape = [0, 0]
    if buffer_type == OpenCLBufferType.CONV2D_FILTER:
        mace_check(len(shape) == 4, "Conv2D filter buffer should be 4D")
        image_shape[0] = shape[1]
        image_shape[1] = shape[2] * shape[3] * roundup_div4(shape[0])
    elif buffer_type == OpenCLBufferType.IN_OUT_CHANNEL:
        mace_check(len(shape) == 2 or len(shape) == 4,
                   "input/output buffer should be 2D|4D")
        if len(shape) == 4:
            image_shape[0] = roundup_div4(shape[3]) * shape[2]
            image_shape[1] = shape[0] * shape[1]
        elif len(shape) == 2:
            image_shape[0] = roundup_div4(shape[1])
            image_shape[1] = shape[0]
    elif buffer_type == OpenCLBufferType.ARGUMENT:
        mace_check(len(shape) == 1,
                   "Argument buffer should be 1D not " + str(shape))
        image_shape[0] = roundup_div4(shape[0])
        image_shape[1] = 1
    elif buffer_type == OpenCLBufferType.IN_OUT_HEIGHT:
        if len(shape) == 4:
            image_shape[0] = shape[2] * shape[3]
            image_shape[1] = shape[0] * roundup_div4(shape[1])
        elif len(shape) == 2:
            image_shape[0] = shape[0]
            image_shape[1] = roundup_div4(shape[1])
    elif buffer_type == OpenCLBufferType.IN_OUT_WIDTH:
        mace_check(len(shape) == 4, "Input/output buffer should be 4D")
        image_shape[0] = roundup_div4(shape[2]) * shape[3]
        image_shape[1] = shape[0] * shape[1]
    elif buffer_type == OpenCLBufferType.WINOGRAD_FILTER:
        mace_check(len(shape) == 4, "Winograd filter buffer should be 4D")
        image_shape[0] = roundup_div4(shape[1])
        image_shape[1] = (shape[0] * (winograd_blk_size + 2)
                          * (winograd_blk_size + 2))
    elif buffer_type == OpenCLBufferType.DW_CONV2D_FILTER:
        mace_check(len(shape) == 4, "Winograd filter buffer should be 4D")
        image_shape[0] = shape[0] * shape[2] * shape[3]
        image_shape[1] = roundup_div4(shape[1])
    elif buffer_type == OpenCLBufferType.WEIGHT_HEIGHT:
        mace_check(len(shape) == 4, "Weight buffer should be 4D")
        image_shape[0] = shape[1] * shape[2] * shape[3]
        image_shape[1] = roundup_div4(shape[0])
    elif buffer_type == OpenCLBufferType.WEIGHT_WIDTH:
        mace_check(len(shape) == 4, "Weight buffer should be 4D")
        image_shape[0] = roundup_div4(shape[1]) * shape[2] * shape[3]
        image_shape[1] = shape[0]
    else:
        mace_check(False, "OpenCL Image do not support type "
                   + str(buffer_type))
    return image_shape


def getAutogenIniPath(input_config, config):
    global AUTOGEN_INI_MODIFY_COUNT
    autogen_path = os.path.join(os.path.dirname(input_config),
                            'autogen_' + os.path.basename(input_config))
    if AUTOGEN_INI_MODIFY_COUNT > 0:
        config.read(autogen_path)
    else:
        config.read(input_config)

    return autogen_path


def setAutogenWarning(input_config):
    global AUTOGEN_INI_MODIFY_COUNT
    AUTOGEN_INI_MODIFY_COUNT += 1
    with open(input_config, 'r') as fr:
        input_config_data = fr.readlines()
    with open(input_config, 'w') as fw:
        str_warn = 'Auto generated by SGS_IPU_SDK_v{}. Do not modify manually!'.format(calibrator_custom.__version__)
        str_extn = '=' * len(str_warn)
        fw.write('; {}\n'.format(str_extn))
        fw.write('; {}\n'.format(str_warn))
        fw.write('; {}\n\n'.format(str_extn))
        fw.writelines(input_config_data)


def check_requirements(requirements):
    requirements = requirements.split(',')
    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):
            s += f'"{r}" '
            n += 1
    if n > 0:
        s = f'requirement {s} not found!'
        print('\033[31mUpgrade the following python packages manually, or use docker 1.7!\033[0m')
        print(s)
        sys.exit(0)

