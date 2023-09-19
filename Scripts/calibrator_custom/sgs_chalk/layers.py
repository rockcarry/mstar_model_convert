# -*- coding: utf-8 -*-

from .sgs_builder import Tensor
from .sgs_builder import Operator
from .sgs_builder import Model
from .sgs_builder import BUILDER
from . import chalk_common
from third_party import tflite
from third_party.python import flatbuffers
import numpy as np
import math
import functools
import pdb


@chalk_common.platform_register
def Add(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns x + y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:

        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Add(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:

        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Add(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.AddOptions.AddOptionsStart(BUILDER)
    tflite.AddOptions.AddOptionsAddFusedActivationFunction(BUILDER,0)
    tflite.AddOptions.AddOptionsAddPotScaleInt16(BUILDER,1)
    add_options = tflite.AddOptions.AddOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Add')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().ADD,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().AddOptions,builtin_options=add_options)
    return out0

@chalk_common.platform_register
def Sub(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns x - y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Sub(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Sub(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.SubOptions.SubOptionsStart(BUILDER)
    tflite.SubOptions.SubOptionsAddFusedActivationFunction(BUILDER,0)
    tflite.SubOptions.SubOptionsAddPotScaleInt16(BUILDER,1)
    sub_options = tflite.SubOptions.SubOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Sub')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().SUB,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SubOptions, builtin_options=sub_options)
    return out0

@chalk_common.platform_register
def Mul(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns x * y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Mul(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Mul(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Mul(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.MulOptions.MulOptionsStart(BUILDER)
    tflite.MulOptions.MulOptionsAddFusedActivationFunction(BUILDER,0)
    mul_options = tflite.MulOptions.MulOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Mul')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().MUL,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().MulOptions,builtin_options=mul_options)
    return out0

@chalk_common.platform_register
def TopKV2(x, k=1, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Finds values and indices of the `k` largest entries for the last dimension.

    If the input is a vector (rank=1), finds the `k` largest entries in the vector
    and outputs their values and indices as vectors.  Thus `values[j]` is the
    `j`-th largest entry in `input`, and its index is `indices[j]`.

    For matrices (resp. higher rank input), computes the top `k` entries in each
    row (resp. vector along the last dimension).  Thus,

        values.shape = indices.shape = input.shape[:-1] + [k]

    If two elements are equal, the lower-index element appears first.

    Args:
    ```text
    x: 1-D or higher `Tensor` with last dimension at least `k`.
    k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
       dimension (along each row for matrices).
    name: Optional name for the operation.
    ```

    Returns:
    ```text
    values: The `k` largest elements along each last dimensional slice.
    indices: The indices of `values` within the last dimension of `input`.
    ```


    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        output_list = sgs_chalk.TopKV2(in0,name=['values','indices'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    tflite.TopKV2Options.TopKV2OptionsStart(BUILDER)
    TopKV2_options = tflite.TopKV2Options.TopKV2OptionsEnd(BUILDER)
    #cal output shape
    output_shape = np.array(x.shape).tolist()[:-1] + [k]
    output_tensor_list = []
    for i in range(2):
        out0 = Tensor(shape=output_shape, name=name[i], bit=bit, minimum=minimum, maximum=maximum, prefix=name[i])
        output_tensor_list.append(out0)
    k = Tensor(data=np.array([k]).astype('int32'),  bit=32, prefix='k')
    Operator([x,k], output_tensor_list, tflite.BuiltinOperator.BuiltinOperator().TOPK_V2,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().TopKV2Options,builtin_options=TopKV2_options)
    return output_tensor_list

@chalk_common.platform_register
def Round(x, k=1, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Rounds the values of a tensor to the nearest integer, element-wise.

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor` of same shape and type as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Round(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='round')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().ROUND)
    return out0

@chalk_common.platform_register
def Sqrt(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes square root of x element-wise.
    I.e., (y = sqrt{x} = x^{1/2}).

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor` of same shape and type as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Sqrt(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='sqrt')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().SQRT)
    return out0

@chalk_common.platform_register
def Abs(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes the absolute value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor` of same shape and type as `x`.
    ```


    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Abs(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    tflite.AbsOptions.AbsOptionsStart(BUILDER)
    abs_options = tflite.AbsOptions.AbsOptionsEnd(BUILDER)
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Abs')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().ABS,builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().AbsOptions,builtin_options=abs_options)
    return out0

@chalk_common.platform_register
def Reciprocal(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes the Reciprocal value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor` of same shape and type as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Reciprocal(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    cus_options = [(b"numeratort",1,"int")]
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Reciprocal')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'Reciprocal',
            builtin_options_type=None,builtin_options=None,custom_options=cus_options)
    return out0

@chalk_common.platform_register
def Fullyconnected(x, weight, bias = None, Activation = 'NONE', name=None, bit=16, minimum=None, maximum=None):
    r"""
    `fully_connected` creates a variable called `weights`, representing a fully
    connected weight matrix, which is multiplied by the `inputs` to produce a
    `Tensor` of hidden units.

    Args:
    ```text
    x : input tensor of shape 2 dims
    weight: filters of shape 2 dims
    bias: optional bias tensor of shape(out_channels).Default:None
    Activation: only support NONE/RELU/RELU_N1_TO_1/RELU6
    ```

    Examples:
    ```python
    case0:
        input1 = np.zeros((1000,112), dtype=np.float32)
        in0 = sgs_chalk.Input((3,112),name='input0')
        out0 = sgs_chalk.Fullyconnected(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((1000,112), dtype=np.float32)
        in0 = sgs_chalk.Input((3,112),name='input0')
        out0 = sgs_chalk.Fullyconnected(in0,input1,bias = None, Activation = 'RELU',name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if len(x.shape) != 2:
        raise ValueError('only support 2dim input !!!')
    if len(weight.shape) != 2:
        raise ValueError('only support 2dim weight !!!')

    if x.shape[1] != weight.shape[1]:
        raise ValueError('x.shape[1] and weight.shape[1] must be same!!!')

    if isinstance(weight, np.ndarray):
        if weight.astype != 'float32':
            weight = weight.astype('float32')
        weight = Tensor(data=weight,name='weight')
    elif isinstance(weight, Tensor):
        weight = weight
    else:
        raise ValueError('Not support this type:', type(weight))

    if bias == None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
        bias = Tensor(data=bias, name='bias', bit=bit, prefix='bias')
    if isinstance(bias, np.ndarray):
        bias = Tensor(data=bias,name='bias')
    elif isinstance(bias, Tensor):
        bias = bias
    else:
        raise ValueError('Not support this type:', type(bias))

    if Activation == 'NONE':
        fusedActivationFunction = 0
    elif Activation == 'RELU':
        fusedActivationFunction = 1
    elif Activation == 'RELU_N1_TO_1':
        fusedActivationFunction = 2
    elif Activation == 'RELU6':
        fusedActivationFunction = 3
    else:
        raise ValueError('Not support this activation !!!')

    tflite.FullyConnectedOptions.FullyConnectedOptionsStart(BUILDER)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddFusedActivationFunction(BUILDER, fusedActivationFunction)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddWeightsFormat(BUILDER, 0)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddKeepNumDims(BUILDER, 0)
    tflite.FullyConnectedOptions.FullyConnectedOptionsAddAsymmetricQuantizeInputs(BUILDER, 0)
    FullyConnect_options = tflite.FullyConnectedOptions.FullyConnectedOptionsEnd(BUILDER)

    out0 = Tensor(shape=[x.shape[0],weight.shape[0]], name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='fullyconnected')
    Operator([x,weight,bias], [out0], tflite.BuiltinOperator.BuiltinOperator().FULLY_CONNECTED,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().FullyConnectedOptions,builtin_options=FullyConnect_options)
    return out0

@chalk_common.platform_register
def Unpack(x, axis, num=None, name=None, bit=16, minimum=None, maximum=None):
    r"""Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

    Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
    If `num` is not specified (the default), it is inferred from `value`'s shape.
    If `value.shape[axis]` is not known, `ValueError` is raised.

    For example, given a tensor of shape `(A, B, C, D)`;

    If `axis == 0` then the i'th tensor in `output` is the slice
      `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
      (Note that the dimension unpacked along is gone, unlike `split`).

    If `axis == 1` then the i'th tensor in `output` is the slice
      `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
    Etc.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    axis: An `int`. The axis to unstack along. Defaults to the first dimension.
        Negative values wrap around, so the valid range is `[-R, R)
    num:  An `int`. The length of the dimension `axis`. Automatically inferred if
        `None` (the default).
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    The list of `Tensor` objects unstacked from `value`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,4,3),name='input0')
        output_list = sgs_chalk.Unpack(in0,axis=2,name=['output0','output1','output2'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)

    case1:
        in0 = sgs_chalk.Input((28,4,3),name='input0')
        output_list = sgs_chalk.Unpack(in0,axis=1,name=['output0','output1','output2','output3'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    axis_value = axis if axis >= 0 else len(x.shape) + axis
    if num == None:
        num = x.shape[axis_value]
    else:
        if num != x.shape[axis_value]:
            raise ValueError('Dimension must be {} but is {} for unpack with input shapes: {}'.format(num,x.shape[axis_value],x.shape))

    tflite.UnpackOptions.UnpackOptionsStart(BUILDER)
    tflite.UnpackOptions.UnpackOptionsAddNum(BUILDER, num)
    tflite.UnpackOptions.UnpackOptionsAddAxis(BUILDER, axis_value)
    Unpack_options = tflite.UnpackOptions.UnpackOptionsEnd(BUILDER)

    #cal output shape
    output_shape = x.shape
    output_shape = np.array(output_shape).tolist()
    del(output_shape[axis])

    output_tensor_list = []
    for i in range(x.shape[axis]):
        out0 = Tensor(shape=output_shape, name=name[i], bit=bit, minimum=minimum, maximum=maximum, prefix='Unpack:'+str(i))
        output_tensor_list.append(out0)
    Operator([x], output_tensor_list, tflite.BuiltinOperator.BuiltinOperator().UNPACK,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().UnpackOptions,builtin_options=Unpack_options)
    return output_tensor_list

@chalk_common.platform_register
def Pack(values, axis, name=None, bit=16, minimum=None, maximum=None):
    r"""Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

    Packs the list of tensors in `values` into a tensor with rank one higher than
    each tensor in `values`, by packing them along the `axis` dimension.
    Given a list of length `N` of tensors of shape `(A, B, C)`;

    if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
    if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
    Etc.

    Args:
    ```text
    x: A list of `Tensor` objects with the same shape and type. Must be Variable Tensor.
    axis: An `int`. The axis to stack along. Defaults to the first dimension.
          Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,1),name='input0')
        in1 = sgs_chalk.Input((28,512,1),name='input1')
        in2 = sgs_chalk.Input((28,512,1),name='input2')
        x = [in0, in1, in2]
        out0 = sgs_chalk.Pack(x,axis=2,name='output')
        model = sgs_chalk.Model(x,out0)
        model.save('test.sim')
    ```
    """
    if not isinstance(values, list) and not isinstance(values, tuple):
        raise ValueError('values of Pack only supports list or tuple')
    values_1 = np.array(values).tolist()
    if axis >= len(values_1[0].shape):
        raise ValueError('axis error!!!')
    axis_value = axis if axis >= 0 else len(values_1[0].shape) + axis
    for i in range(len(values_1)):
        if not isinstance(values_1[i], Tensor):
            raise ValueError('values musr be a list or a tuple of tensor!!')

    #cal output shape
    output_shape_aixs = 0
    output_shape = values_1[0].shape
    output_shape = np.array(output_shape).tolist()
    output_shape.insert(axis_value,len(values))

    tflite.PackOptions.PackOptionsStart(BUILDER)
    tflite.PackOptions.PackOptionsAddValuesCount(BUILDER, len(values))
    tflite.PackOptions.PackOptionsAddAxis(BUILDER, axis_value)
    pack_options = tflite.PackOptions.PackOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Pack')
    Operator(values, [out0], tflite.BuiltinOperator.BuiltinOperator().PACK,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().PackOptions,builtin_options=pack_options)
    return out0

@chalk_common.platform_register
def Slice(x, begin, size, name=None, bit=16, minimum=None, maximum=None):
    r"""Extracts a slice from a tensor.

    This operation extracts a slice of size `size` from a tensor `input` starting
    at the location specified by `begin`. The slice `size` is represented as a
    tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
    of `input` that you want to slice. The starting location (`begin`) for the
    slice is represented as an offset in each dimension of `input`. In other
    words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
    want to slice from.


    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    begin: const Tensor
    size: const Tensor
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `size`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Slice(in0,[0,0],[28,512],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Slice(in0,[0,0],[14,510],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if not isinstance(begin, list):
        raise ValueError('begin of Slice only supports list')
    if not isinstance(size, list) :
        raise ValueError('new_shape of Slice only supports list ')

    begin_1 = np.array(begin)
    size_1 = np.array(size)
    if len(x.shape) != len(begin_1):
        raise ValueError('length of x.shape must equal length of begin.shape')
    if len(x.shape) != len(size_1):
        raise ValueError('length of x.shape must equal length of size.shape')

    for i in range(len(x.shape)):
        if x.shape[i] < begin_1[i] + size_1[i]:
             raise ValueError('wrong begin or size parameter,x.shape[i] >= begin[i] + size[i]!!!!')

    tflite.SliceOptions.SliceOptionsStart(BUILDER)
    slice_options = tflite.SliceOptions.SliceOptionsEnd(BUILDER)

    begin = Tensor(data=np.array(begin).astype('int32'),  bit=bit, prefix='begin')
    size = Tensor(data=np.array(size).astype('int32'),  bit=bit, prefix='size')
    out0 = Tensor(shape=list(size_1), name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Slice')
    Operator([x, begin, size], [out0], tflite.BuiltinOperator.BuiltinOperator().SLICE,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SliceOptions,builtin_options=slice_options)
    return out0

@chalk_common.platform_register
def StridedSlice(x, begin, end, strides, name=None, bit=16, minimum=None, maximum=None):
    r"""Extracts a strided slice of a tensor (generalized python array indexing).

    Roughly speaking, this op extracts a slice of size `(end-begin)/stride`
    from the given `input_` tensor. Starting at the location specified by `begin`
    the slice continues by adding `stride` to the index until all dimensions are
    not less than `end`.
    Note that a stride can be negative, which causes a reverse slice.

    Given a Python slice `input[spec0, spec1, ..., specn]`,
    this function will be called as follows.

    `begin`, `end`, and `strides` will be vectors of length n.
    n in general is not equal to the rank of the `input_` tensor.


    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    begin: const Tensor
    size: const Tensor
    begin_mask: An `int32` mask.
    end_mask: An `int32` mask.
    ellipsis_mask: An `int32` mask.
    new_axis_mask: An `int32` mask.
    shrink_axis_mask: An `int32` mask.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`. (end-begin)/stride
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.StridedSlice(in0,[1,0,0],[28,512,4],[1,2,1],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if not isinstance(begin, list):
        raise ValueError('begin of StridedSlice only supports list')
    if not isinstance(end, list) :
        raise ValueError('end of StridedSlice only supports list ')
    if not isinstance(strides, list) :
        raise ValueError('strides of StridedSlice only supports list ')

    begin_1 = np.array(begin)
    end_1 = np.array(end)
    strides_1 = np.array(strides)

    if len(x.shape) != len(begin_1):
        raise ValueError('length of x.shape must equal length of begin.shape')
    if len(x.shape) != len(end_1):
        raise ValueError('length of x.shape must equal length of size.shape')
    if len(x.shape) != len(strides_1):
        raise ValueError('length of x.shape must equal length of strides.shape')

    #cal output shape
    output_shape = []
    for i in range(len(x.shape)):
        shape = (end[i] - begin[i]) // strides[i]
        output_shape.append(shape)

    begin = Tensor(data=np.array(begin).astype('int32'),  bit=bit, prefix='begin')
    end = Tensor(data=np.array(end).astype('int32'),  bit=bit, prefix='end')
    strides = Tensor(data=np.array(strides).astype('int32'),  bit=bit, prefix='strides')

    beginMask=0
    endMask=0
    ellipsisMask=0
    newAxisMask=0
    shrinkAxisMask=0
    tflite.StridedSliceOptions.StridedSliceOptionsStart(BUILDER)
    tflite.StridedSliceOptions.StridedSliceOptionsAddBeginMask(BUILDER, beginMask)
    tflite.StridedSliceOptions.StridedSliceOptionsAddEndMask(BUILDER, endMask)
    tflite.StridedSliceOptions.StridedSliceOptionsAddEllipsisMask(BUILDER, ellipsisMask)
    tflite.StridedSliceOptions.StridedSliceOptionsAddNewAxisMask(BUILDER, newAxisMask)
    tflite.StridedSliceOptions.StridedSliceOptionsAddShrinkAxisMask(BUILDER, shrinkAxisMask)
    StridedSlice_options = tflite.StridedSliceOptions.StridedSliceOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='StridedSlice')
    Operator([x, begin, end, strides], [out0], tflite.BuiltinOperator.BuiltinOperator().STRIDED_SLICE,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().StridedSliceOptions,builtin_options=StridedSlice_options)
    return out0

@chalk_common.platform_register
def Concatenation(values, axis=0, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Concatenates the list of tensors `values` along dimension `axis`.  If
    `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
    result has shape [D0, D1, ... Raxis, ...Dn] where  Raxis = sum(Daxis(i))
    That is, the data from the input tensors is joined along the `axis`
    dimension.
    The number of dimensions of the input tensors must match, and all dimensions
    except `axis` must be equal.

    Args:
    ```text
    values: A list of `Tensor` objects or a single `Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
        in the range `[-rank(values), rank(values))`. As in Python, indexing for
        axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
        to `axis`-th dimension. And negative axis refers to `axis +
        rank(values)`-th dimension.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,1),name='input0')
        in1 = sgs_chalk.Input((28,512,2),name='input1')
        in2 = sgs_chalk.Input((28,512,3),name='input2')
        x = [in0, in1, in2]
        out0 = sgs_chalk.Concatenation(x,axis=2,name='output')
        model = sgs_chalk.Model(x,out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,1,512),name='input0')
        in1 = sgs_chalk.Input((28,2,512),name='input1')
        in2 = sgs_chalk.Input((28,3,512),name='input2')
        x = [in0, in1, in2]
        out0 = sgs_chalk.Concatenation(x,axis=1,name='output')
        model = sgs_chalk.Model(x,out0)
        model.save('test.sim')
    ```
    """
    if not isinstance(values, list) and not isinstance(values, tuple):
        raise ValueError('Input(0) `values` only supports list or tuple')
    values_1 = np.array(values).tolist()
    axis_value = axis if axis >= 0 else len(values_1[0].shape) + axis
    if axis_value >= len(values_1[0].shape):
        raise ValueError('axis error!!!')
    for i in range(len(values_1)):
        if not isinstance(values_1[i], Tensor):
            raise ValueError('`values` must be a list or a tuple of tensor!!')
    shape_str = ''
    for value in values:
        shape_str += '['
        shape_str += ', '.join(str(i) for i in value.shape)
        shape_str += '] '
    input_lens = [len(x.shape) for x in values]
    if not np.equal(input_lens, input_lens[0]).all():
        raise ValueError('Concatenate requires all the inputs must have same number of dimensions. Such inputs cannot concatenate: {}'.format(
            shape_str))
    in_shape = values[0].shape
    shape_check = np.equal(in_shape, in_shape)
    for value in values[1:]:
        shape_check = np.equal(shape_check, np.equal(in_shape, value.shape))
    shape_check = shape_check.tolist()
    del shape_check[axis_value]
    if not np.all(shape_check):
        raise ValueError('Concatenate requires all the inputs dimensions except for the concatenation axis must match exactly. Such inputs cannot concatenate at axis={}: {}'.format(
            axis, shape_str))

    #cal output shape
    output_shape_axis = 0
    for value in values:
        output_shape_axis += value.shape[axis_value]
    output_shape = list(values[0].shape)
    output_shape[axis_value] = output_shape_axis

    tflite.ConcatenationOptions.ConcatenationOptionsStart(BUILDER)
    tflite.ConcatenationOptions.ConcatenationOptionsAddAxis(BUILDER, axis_value)
    tflite.ConcatenationOptions.ConcatenationOptionsAddFusedActivationFunction(BUILDER,0)
    concat_options = tflite.ConcatenationOptions.ConcatenationOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Concatenation')
    Operator(values_1, [out0], tflite.BuiltinOperator.BuiltinOperator().CONCATENATION,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ConcatenationOptions,builtin_options=concat_options)
    return out0

@chalk_common.platform_register
def TFLite_Detection_NMS(x1,y1,x2,y2,confidence,score,max_score=None,mode='YOLO',max_detections=100,nms_score_threshold=0.00499999989,nms_iou_threshold=0.449999988,
                         num_classes=1,clip=1,is_need_index=False,name=None, bit=16, minimum=None, maximum=None):
    r"""
    sigmastar postprocess nms

    Args:
    ```text
    max_detection: max number pf output dectection bboxes
    num_classes:number of classes
    is_need_index : outputs include index or not
    ```

    Examples:
    ```python
    case0:
        bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
        confidence_tensor = sgs_chalk.Logistic(output_list[4],name="confidence_tensor")
        SGS_score0 = sgs_chalk.Logistic(output_list[5],name="score0_tensor")
        SGS_score1 = sgs_chalk.Mul(confidence_tensor,SGS_score0,name="SGS_score1")
        out1 = sgs_chalk.TFLite_Detection_NMS(bosdecoder_output_list[0],bosdecoder_output_list[1],bosdecoder_output_list[2],
                                          bosdecoder_output_list[3],confidence_tensor,SGS_score1,output_list[6],mode='YOLO',
                                          max_detections=100,nms_score_threshold=0.4,
                                          nms_iou_threshold=0.45,num_classes=80,is_need_index=False)
    case1:
        bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
        mbox_conf_softmax = sgs_chalk.Input(model_config["input_shape"][1], name = model_config["input"][1])
        postprocess_max_output_list = sgs_chalk.PostProcess_Max(mbox_conf_softmax,scores_lengh=21,skip=1)
        out1 = sgs_chalk.TFLite_Detection_NMS(bosdecoder_output_list[0],bosdecoder_output_list[1],bosdecoder_output_list[2],
                                          bosdecoder_output_list[3],postprocess_max_output_list[0],postprocess_max_output_list[1],
                                          mode='SSD',max_detections=10,nms_score_threshold=0.01,
                                          nms_iou_threshold=0.45,num_classes=20,is_need_index=False)
    ```
    """

    if not isinstance(x1, Tensor):
        raise ValueError('please input x1 tensor!!')
    if not isinstance(y1, Tensor):
        raise ValueError('please input y1 tensor!!')
    if not isinstance(x2, Tensor):
        raise ValueError('please input x2 tensor!!')
    if not isinstance(y2, Tensor):
        raise ValueError('please input y2 tensor!!')
    if not isinstance(confidence, Tensor):
        raise ValueError('please input confidence tensor!!')
    if not isinstance(score, Tensor):
        raise ValueError('please input score tensor!!')

    if mode =='YOLO':
        if not isinstance(max_score, Tensor):
            raise ValueError('please input max_score tensor!!')

    if mode =='SSD':
        if isinstance(max_score, Tensor):
            raise ValueError('SSD no max_score tensor!!')
        if x1.shape[1] > 24576:
            raise ValueError('SSD  max bbox input is 24576!!')

    if not is_need_index:
        output_detection_boxes_index_idx = -1
    else:
        output_detection_boxes_index_idx = 4

    input_list=[]
    if mode =='YOLO':
        input_class_idx = 6
        input_score_idx = 5
        input_confidence_idx = 4
        input_list=[x1,y1,x2,y2,confidence,score,max_score]
        bmax_score=1
    elif mode == 'SSD':
        input_class_idx = 5
        input_score_idx = 4
        input_confidence_idx = -1
        input_list=[x1,y1,x2,y2,confidence,score]
        bmax_score=0
    else:
        raise ValueError('only support YOLO & SSD!!')

    nms_cus_options = [(b"input_coordinate_x1",0,"int"),
                        (b"input_coordinate_y1",1,"int"),
                        (b"input_coordinate_x2",2,"int"),
                        (b"input_coordinate_y2",3,"int"),
                        (b"input_class_idx",input_class_idx,"int"),
                        (b"input_score_idx",input_score_idx,"int"),
                        (b"input_confidence_idx",input_confidence_idx,"int"),
                        (b"input_facecoordinate_idx",-1,"int"),
                        (b"output_detection_boxes_idx",0,"int"),
                        (b"output_detection_classes_idx",1,"int"),
                        (b"output_detection_scores_idx",2,"int"),
                        (b"output_num_detection_idx",3,"int"),

                        (b"output_detection_boxes_index_idx",output_detection_boxes_index_idx,"int"),
                        (b"max_detections",max_detections,"int"),
                        (b"nms_score_threshold",nms_score_threshold,"float"),
                        (b"nms_iou_threshold",nms_iou_threshold,"float"),
                        (b"nms",0,"float"),
                        (b"max_classes_per_detection",1,"int"),
                        (b"detections_per_class",1,"int"),
                        (b"num_classes_with_background",1,"int"),

                        (b"clip",clip,"float"),
                        (b"num_classes",num_classes,"int"),
                        (b"bmax_score",bmax_score,"int"),
                        (b"offline",0,"int"),
                        ]


    out_shapes = [[1,max_detections,4],[1,max_detections],[1,max_detections],[1],[1,max_detections]]

    out0 = Tensor(shape = out_shapes[0], name="detectionBoxes")
    out1 = Tensor(shape = out_shapes[1], name="detectionClasses")
    out2 = Tensor(shape = out_shapes[2], name="detectionScores")
    out3 = Tensor(shape = out_shapes[3], name="numDetections")

    if is_need_index == False:
        nms_output = [out0,out1,out2,out3]
    else:
        out4 = Tensor(shape = out_shapes[4], name="detectionIndex")
        nms_output = [out0,out1,out2,out3,out4]
    Operator(input_list, nms_output, tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'TFLite_Detection_NMS',
            builtin_options_type=None,builtin_options=None,custom_options=nms_cus_options)
    return nms_output

@chalk_common.platform_register
def PostProcess_Unpack(x, mode='YOLO',num_classes=0, name=None, output_shapes=None, bit=16, minimum=None, maximum=None):
    r"""
    sigmastar postprocess unpack

    Args:
    ```text
    num_classes: number of classes
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,3249,6), name = 'x')
        unpack_out_tensors1 = []
        for i in range(4):
            unpack_out_tensors1.append("SGS_unpack1_"+str(i))
        output_list = sgs_chalk.PostProcess_Unpack(in0,scores_lengh=80,name=unpack_out_tensors1)
        model = sgs_chalk.Model([in0],output_list)
        #model.save('test.sim')
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """

    if mode == 'YOLO':
        unpack_length = 7
        confidence_offset = 4
        confidence_lengh = 1
        scores_offset = 5
        max_score = 1
    elif mode == 'SSD':
        unpack_length = 4
        confidence_offset = 0
        confidence_lengh = 0
        scores_offset = 0
        max_score = 0
    else:
        raise ValueError('only support YOLO & SSD!!')

    if mode =='SSD':
        if x.shape[1] > 24576:
            raise ValueError('SSD  max bbox input is 24576!!')

    PostProcess_Unpack_options = [(b"x_offset",0,"int"),
                                (b"x_lengh",1,"int"),
                                (b"y_offset",1,"int"),
                                (b"y_lengh",1,"int"),
                                (b"w_offset",2,"int"),
                                (b"w_lengh",1,"int"),
                                (b"h_offset",3,"int"),
                                (b"h_lengh",1,"int"),
                                (b"confidence_offset",confidence_offset,"int"),
                                (b"confidence_lengh",confidence_lengh,"int"),
                                (b"scores_offset",scores_offset,"int"),
                                (b"scores_lengh",num_classes,"int"),
                                (b"max_score",max_score,"int")]

    #cal output shape
    output_shape = x.shape
    output_shape = np.array(output_shape).tolist()
    del(output_shape[-1])

    output_tensor_list = []
    if mode == 'YOLO':
        for i in range(unpack_length):
            if i == 4:
                out0 = Tensor(shape=output_shape, name='confidence', bit=bit, minimum=minimum, maximum=maximum, prefix='confidence')
            elif i == 5:
                out0 = Tensor(shape=output_shape, name='score', bit=bit, minimum=minimum, maximum=maximum, prefix='score')
            elif i == 6:
                out0 = Tensor(shape=output_shape, name='max_score', bit=bit, minimum=minimum, maximum=maximum, prefix='max_score')
            else:
                out0 = Tensor(shape=output_shape, name=name[i], bit=bit, minimum=minimum, maximum=maximum, prefix='Unpack:'+str(i))
            output_tensor_list.append(out0)
    elif mode == 'SSD':
        for i in range(unpack_length):
            out0 = Tensor(shape=output_shape, name=name[i], bit=bit, minimum=minimum, maximum=maximum, prefix='Unpack:'+str(i))
            output_tensor_list.append(out0)
    else:
        raise ValueError('only support YOLO & SSD!!')

    Operator([x], output_tensor_list, tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'PostProcess_Unpack',
            builtin_options_type=None,builtin_options=None,custom_options=PostProcess_Unpack_options)
    return output_tensor_list

@chalk_common.platform_register
def PostProcess_Max(x, num_classes=21, is_skip_background=None, name=None, bit=16, minimum=None, maximum=None):
    r"""
    sigmastar postprocess max

    Args:
    ```text
    num_classes: number of classes
    is_skip_background: please note background must be the first one
    ```

    Examples:
    ```python
    case0:
        mbox_conf_softmax = sgs_chalk.Tensor(shape=model_config["input_shape"][1],name = model_config["input"][1])
        cus_options = [(b"scores_lengh",21,"int"),
                    (b"skip",1,"int")]
        postprocess_max_output_list = sgs_chalk.PostProcess_Max(mbox_conf_softmax,num_classes=21,skip=1)
        model = sgs_chalk.Model([mbox_conf_softmax],postprocess_max_output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if is_skip_background != None:
        PostProcess_Max_options = [(b"scores_lengh",num_classes,"int"),
                                    (b"skip",is_skip_background,"int")]
    else:
        PostProcess_Max_options = [(b"scores_lengh",num_classes,"int")]

    output_shape = [x.shape[0],x.shape[1]]

    out0 = Tensor(shape = output_shape, name="SGS_PostProcess_Max")
    out1 = Tensor(shape = output_shape, name="SGS_PostProcess_Classes")
    max_output_list = [out0,out1]

    Operator([x], max_output_list, tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'PostProcess_Max',
            builtin_options_type=None,builtin_options=None,custom_options=PostProcess_Max_options)
    return max_output_list

@chalk_common.platform_register
def BoxDecoder(config,unpacked_box):
    r"""
    sigmastar postprocess BoxDecoder

    Args:
    ```text
    unpacked_box: a list of tensors which are unpacked
    ```

    Return:
    ```text
    a list of tensors decoded
    ```

    Examples:
    ```python
    case0:
    box_num = 9
    side_x = 19
    side_y = 19
    ppw = anchor.ones(3249)
    px = anchor.index_div_linear(1,1,0,box_num ,side_x,side_y)
    pph = anchor.ones(3249)
    py = anchor.index_div_linear(1,1,0,side_x*box_num,side_y,1)
    pw = anchor.ones(3249)
    ph = anchor.ones(3249)

    sx = anchor.ns(3249,1.0/19)
    sy = anchor.ns(3249,1.0/19)

    biases= [[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5]]
    sw = [x[0]/(2*19) for x in biases ]*(19*19)
    sh = [x[1]/(2*19) for x in biases ]*(19*19)
    config = {"shape" : [1,3249],
          "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'x_scale'
          "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'y_scale'
          "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'w_scale'
          "th_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'h_scale'
          "x_scale" : 0.1,
          "y_scale" : 0.1,
          "w_scale" : 1,
          "h_scale" : 1,
          "anchor_selector" : "constant",
          "pw" : pw,
          "ph" : ph,
          "pw_func" : (None,None),
          "ph_func" : (None,None),
          "ppw" : ppw,
          "px" : px,
          "pph" : pph,
          "py" : py,
          "sx" : sx,
          "sy" : sy,
          "sw" : sw,
          "sh" : sh
          }
    in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])
    unpack_out_tensors1 = []
    for i in range(7):
        unpack_out_tensors1.append("SGS_unpack1_"+str(i))
    output_list = sgs_chalk.PostProcess_Unpack(in0,name=unpack_out_tensors1)

    bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
    model = sgs_chalk.Model([in0],bosdecoder_output_list)
    model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """

    tx_out_tensors = []
    tx_in_tensors = []
    tx_in_tensors.append(unpacked_box[0])
    if config["tx_func"][1] is None:
        tx_tensor = Tensor(shape=config["shape"],name = "tx_tensor")
    elif config["tx_func"][1] == "x_scale":
        tx_tensor = Tensor(shape=config["shape"],name = "tx_tensor")
        x_scale = Tensor(data=np.array([config["x_scale"]]).astype(np.float32),name = "x_scale")
        tx_in_tensors.append(x_scale)
    else:
        None
    tx_out_tensors.append(tx_tensor)
    Operator(tx_in_tensors,tx_out_tensors, config["tx_func"][0])

    ty_out_tensors = []
    ty_in_tensors = []
    ty_in_tensors.append(unpacked_box[1])
    if config["ty_func"][1] is None:
        ty_tensor = Tensor(shape=config["shape"],name = "ty_tensor")
    elif config["ty_func"][1] == "y_scale":
        ty_tensor = Tensor(shape=config["shape"],name = "ty_tensor")
        y_scale = Tensor(data=np.array([config["y_scale"]]).astype(np.float32),name = "y_scale")
        ty_in_tensors.append(y_scale)
    else:
        None
    ty_out_tensors.append(ty_tensor)
    Operator(ty_in_tensors,ty_out_tensors, config["ty_func"][0])

    tw_out_tensors = []
    tw_in_tensors = []
    if(config["tw_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        tw_tensor = Reshape(unpacked_box[2],config["shape"],name="tw_tensor")
    else:
        tw_in_tensors.append(unpacked_box[2])
        if config["tw_func"][1] is None:
            tw_tensor = Tensor(shape=config["shape"],name = "tw_tensor")
        elif config["tw_func"][1] == "w_scale":
            tw_tensor = Tensor(shape=config["shape"],name = "tw_tensor")
            w_scale = Tensor(data=np.array([config["w_scale"]]).astype(np.float32),name = "w_scale")
            tw_in_tensors.append(w_scale)
        else:
            None
        tw_out_tensors.append(tw_tensor)
        Operator(tw_in_tensors,tw_out_tensors, config["tw_func"][0])


    th_out_tensors = []
    th_in_tensors = []
    if(config["th_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        th_tensor = Reshape(unpacked_box[3],config["shape"],name="th_tensor")

    else:
        th_in_tensors.append(unpacked_box[3])
        if config["th_func"][1] is None:
            th_tensor = Tensor(shape=config["shape"],name = "th_tensor")
        elif config["th_func"][1] == "h_scale":
            th_tensor = Tensor(shape=config["shape"],name = "th_tensor")
            h_scale = Tensor(data=np.array([config["h_scale"]]).astype(np.float32),name = "h_scale")
            th_in_tensors.append(w_scale)
        else:
            None

        th_out_tensors.append(th_tensor)
        Operator(th_in_tensors,th_out_tensors, config["th_func"][0])


    if config["anchor_selector"] == "constant":
        ph_tensor = Tensor(data=np.array(config["ph"]).astype(np.float32),name = "ph_tensor")
        pw_tensor = Tensor(data=np.array(config["pw"]).astype(np.float32),name = "pw_tensor")

    else:
        ph_out_tensors = []
        ph_in_tensors = []
        ph_in_tensors.append(unpacked_box[3])
        ph_tensor = Tensor(shape=config["shape"],name = "ph_tensor")
        ph_out_tensors.append(ph_tensor)
        Operator(ph_in_tensors,th_out_tensors, config["ph_func"][0])

        pw_out_tensors = []
        pw_in_tensors = []
        pw_in_tensors.append(unpacked_box[2])
        pw_tensor = Tensor(shape=config["shape"],name = "pw_tensor")
        pw_out_tensors.append(pw_tensor)
        Operator(pw_in_tensors,pw_out_tensors, config["pw_func"][0])

    ppw_tensor = Tensor(data=np.array(config["ppw"]).astype(np.float32),name = "ppw_tensor")
    px_tensor = Tensor(data=np.array(config["px"]).astype(np.float32),name = "px_tensor")
    pph_tensor = Tensor(data=np.array(config["pph"]).astype(np.float32),name = "pph_tensor")
    py_tensor = Tensor(data=np.array(config["py"]).astype(np.float32),name = "py_tensor")

    x_multi0_tensor = Mul(tx_tensor,ppw_tensor,name="x_multi0_tensor")
    x_add_tensor = Add(x_multi0_tensor,px_tensor,name="x_add_tensor")

    sx_tensor = Tensor(data=np.array(config["sx"]).astype(np.float32),name = "sx_tensor")
    sy_tensor = Tensor(data=np.array(config["sy"]).astype(np.float32),name = "sy_tensor")
    sw_tensor = Tensor(data=np.array(config["sw"]).astype(np.float32),name = "sw_tensor")
    sh_tensor = Tensor(data=np.array(config["sh"]).astype(np.float32),name = "sh_tensor")

    x_multi1_tensor = Mul(x_add_tensor,sx_tensor,name="x_multi1_tensor")

    y_multi0_tensor = Mul(ty_tensor,pph_tensor,name="y_multi0_tensor")
    y_add_tensor = Add(y_multi0_tensor,py_tensor,name="y_add_tensor")
    y_multi1_tensor = Mul(y_add_tensor,sy_tensor,name="y_multi1_tensor")

    w_exp_tensor = Exp(tw_tensor,name="w_exp_tensor")
    w_multi0_tensor = Mul(w_exp_tensor,pw_tensor,name="w_multi0_tensor")
    w_multi1_tensor = Mul(w_multi0_tensor,sw_tensor,name="w_multi1_tensor")

    h_exp_tensor = Exp(th_tensor,name="h_exp_tensor")
    h_multi0_tensor = Mul(h_exp_tensor,ph_tensor,name="h_multi0_tensor")
    h_multi1_tensor = Mul(h_multi0_tensor,sh_tensor,name="h_multi1_tensor")

    x1_tensor = Sub(x_multi1_tensor,w_multi1_tensor,name="x1_tensor")
    y1_tensor = Sub(y_multi1_tensor,h_multi1_tensor,name="y1_tensor")
    x2_tensor = Add(x_multi1_tensor,w_multi1_tensor,name="x2_tensor")
    y2_tensor = Add(y_multi1_tensor,h_multi1_tensor,name="y2_tensor")
    bosdecoder_output_list = []
    bosdecoder_output_list = [x1_tensor,y1_tensor,x2_tensor,y2_tensor]

    return bosdecoder_output_list

@chalk_common.platform_register
def BoxDecoder2( config,unpacked_box):
    r"""
    sigmastar postprocess BoxDecoder2

    Args:
    ```text
    unpacked_box: a list of tensors which are unpacked
    ```

    Return:
    ```text
    :return:a list of tensors decoded
    ```

    Examples:
    ```python
    case0:
        box_num = 9
        side_x = 19
        side_y = 19
        ppw = anchor.ones(3249)
        px = anchor.index_div_linear(1,1,0,box_num ,side_x,side_y)
        pph = anchor.ones(3249)
        py = anchor.index_div_linear(1,1,0,side_x*box_num,side_y,1)
        pw = anchor.ones(3249)
        ph = anchor.ones(3249)

        sx = anchor.ns(3249,1.0/19)
        sy = anchor.ns(3249,1.0/19)

        biases= [[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5],[9.5,9.5]]
        sw = [x[0]/(2*19) for x in biases ]*(19*19)
        sh = [x[1]/(2*19) for x in biases ]*(19*19)
        config = {"shape" : [1,3249],
            "tx_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'x_scale'
            "ty_func" : (tflite.BuiltinOperator.BuiltinOperator().LOGISTIC,None),#None or 'y_scale'
            "tw_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'w_scale'
            "th_func" : (tflite.BuiltinOperator.BuiltinOperator().RESHAPE,None),#None or 'h_scale'
            "x_scale" : 0.1,
            "y_scale" : 0.1,
            "w_scale" : 1,
            "h_scale" : 1,
            "anchor_selector" : "constant",
            "pw" : pw,
            "ph" : ph,
            "pw_func" : (None,None),
            "ph_func" : (None,None),
            "ppw" : ppw,
            "px" : px,
            "pph" : pph,
            "py" : py,
            "sx" : sx,
            "sy" : sy,
            "sw" : sw,
            "sh" : sh
            }
        in0 = sgs_chalk.Input(model_config["input_shape"][0], name = model_config["input"][0])
        unpack_out_tensors1 = []
        for i in range(7):
            unpack_out_tensors1.append("SGS_unpack1_"+str(i))
        output_list = sgs_chalk.PostProcess_Unpack(in0,name=unpack_out_tensors1)

        bosdecoder_output_list = sgs_chalk.BoxDecoder(config,output_list)
        model = sgs_chalk.Model([in0],bosdecoder_output_list)
        #model.save('test.sim')
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """

    tx_out_tensors = []
    tx_in_tensors = []

    if(config["tx_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        tx_tensor = Reshape(unpacked_box[0],config["shape"],name="tx_tensor")
    else:
        tx_in_tensors.append(unpacked_box[0])
        if config["tx_func"][1] is None:
            tx_tensor = Tensor(shape=config["shape"],name = "tx_tensor")
        elif config["tx_func"][1] == "x_scale":
            tx_tensor = Tensor(shape=config["shape"],name = "tx_tensor")
            x_scale = Tensor(data=np.array([config["x_scale"]]).astype(np.float32),name = "x_scale")
            tx_in_tensors.append(x_scale)
        else:
            None
        tx_out_tensors.append(tx_tensor)
        Operator(tx_in_tensors,tx_out_tensors, config["tx_func"][0])

    ty_out_tensors = []
    ty_in_tensors = []
    if(config["ty_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        ty_tensor = Reshape(unpacked_box[1],config["shape"],name="ty_tensor")
    else:
        ty_in_tensors.append(unpacked_box[1])
        if config["ty_func"][1] is None:
            ty_tensor = Tensor(shape=config["shape"],name = "ty_tensor")
        elif config["ty_func"][1] == "y_scale":
            ty_tensor = Tensor(shape=config["shape"],name = "ty_tensor")
            y_scale = Tensor(data=np.array([config["y_scale"]]).astype(np.float32),name = "y_scale")
            ty_in_tensors.append(y_scale)
        else:
            None
        ty_out_tensors.append(ty_tensor)
        Operator(ty_in_tensors,ty_out_tensors, config["ty_func"][0])

    tw_out_tensors = []
    tw_in_tensors = []
    if(config["tw_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        tw_tensor = Reshape(unpacked_box[2],config["shape"],name="tw_tensor")
    else:
        tw_in_tensors.append(unpacked_box[2])
        if config["tw_func"][1] is None:
            tw_tensor = Tensor(shape=config["shape"],name = "tw_tensor")
        elif config["tw_func"][1] == "w_scale":
            tw_tensor = Tensor(shape=config["shape"],name = "tw_tensor")
            w_scale = Tensor(data=np.array([config["w_scale"]]).astype(np.float32),name = "w_scale")
            tw_in_tensors.append(w_scale)
        else:
            None
        tw_out_tensors.append(tw_tensor)
        Operator(tw_in_tensors,tw_out_tensors, config["tw_func"][0])


    th_out_tensors = []
    th_in_tensors = []
    if(config["th_func"][0] == tflite.BuiltinOperator.BuiltinOperator().RESHAPE):
        th_tensor = Reshape(unpacked_box[3],config["shape"],name="th_tensor")

    else:
        th_in_tensors.append(unpacked_box[3])
        if config["th_func"][1] is None:
            th_tensor = Tensor(shape=config["shape"],name = "th_tensor")
        elif config["th_func"][1] == "h_scale":
            th_tensor = Tensor(shape=config["shape"],name = "th_tensor")
            h_scale = Tensor(data=np.array([config["h_scale"]]).astype(np.float32),name = "h_scale")
            th_in_tensors.append(w_scale)
        else:
            None

        th_out_tensors.append(th_tensor)
        Operator(th_in_tensors,th_out_tensors, config["th_func"][0])


    if config["anchor_selector"] == "constant":
        ph_tensor = Tensor(data=np.array(config["ph"]).astype(np.float32),name = "ph_tensor")
        pw_tensor = Tensor(data=np.array(config["pw"]).astype(np.float32),name = "pw_tensor")

    else:
        ph_out_tensors = []
        ph_in_tensors = []
        ph_in_tensors.append(unpacked_box[3])
        ph_tensor = Tensor(shape=config["shape"],name = "ph_tensor")
        ph_out_tensors.append(ph_tensor)
        Operator(ph_in_tensors,th_out_tensors, config["ph_func"][0])

        pw_out_tensors = []
        pw_in_tensors = []
        pw_in_tensors.append(unpacked_box[2])
        pw_tensor = Tensor(shape=config["shape"],name = "pw_tensor")
        pw_out_tensors.append(pw_tensor)
        Operator(pw_in_tensors,pw_out_tensors, config["pw_func"][0])

    ppw_tensor = Tensor(data=np.array(config["ppw"]).astype(np.float32),name = "ppw_tensor")
    px_tensor = Tensor(data=np.array(config["px"]).astype(np.float32),name = "px_tensor")
    pph_tensor = Tensor(data=np.array(config["pph"]).astype(np.float32),name = "pph_tensor")
    py_tensor = Tensor(data=np.array(config["py"]).astype(np.float32),name = "py_tensor")

    positive_one_tensor = Tensor(data=np.array([1.0]).astype(np.float32),name = "positive_one_tensor")

    x_multi0_tensor = Mul(tx_tensor,positive_one_tensor,name="x_multi0_tensor")
    x_add_tensor = Add(x_multi0_tensor,px_tensor,name="x_add_tensor")

    sx_tensor = Tensor(data=np.array(config["sx"]).astype(np.float32),name = "sx_tensor")
    sy_tensor = Tensor(data=np.array(config["sy"]).astype(np.float32),name = "sy_tensor")
    sw_tensor = Tensor(data=np.array(config["sw"]).astype(np.float32),name = "sw_tensor")
    sh_tensor = Tensor(data=np.array(config["sh"]).astype(np.float32),name = "sh_tensor")

    x_multi1_tensor = Mul(x_add_tensor,sx_tensor,name="x_multi1_tensor")

    y_multi0_tensor = Mul(ty_tensor,positive_one_tensor,name="y_multi0_tensor")
    y_add_tensor = Add(y_multi0_tensor,py_tensor,name="y_add_tensor")
    y_multi1_tensor = Mul(y_add_tensor,sy_tensor,name="y_multi1_tensor")

    w_sub_tensor = Sub(tw_tensor,positive_one_tensor,name="w_sub_tensor")
    w_sub_reciprocal_tensor = Reciprocal(w_sub_tensor,name="w_sub_reciprocal_tensor")

    negative_one_tensor = Tensor(data=np.array([-1.0]).astype(np.float32),name = "negative_one_tensor")

    w_sub_div_mul_tensor = Mul(w_sub_reciprocal_tensor,negative_one_tensor,name="w_sub_div_mul_tensor")
    w_sub_div_mul_sub_tensor = Sub(w_sub_div_mul_tensor,positive_one_tensor,name="w_sub_div_mul_sub_tensor")
    w_multi0_tensor = Mul(w_sub_div_mul_sub_tensor,pw_tensor,name="w_multi0_tensor")
    w_multi1_tensor = Mul(w_multi0_tensor,sw_tensor,name="w_multi1_tensor")

    h_sub_tensor = Sub(th_tensor,positive_one_tensor,name="h_sub_tensor")
    h_sub_reciprocal_tensor = Reciprocal(h_sub_tensor,name="h_sub_reciprocal_tensor")
    h_sub_div_mul_tensor = Mul(h_sub_reciprocal_tensor,negative_one_tensor,name="h_sub_div_mul_tensor")
    h_sub_div_mul_sub_tensor = Sub(h_sub_div_mul_tensor,positive_one_tensor,name="h_sub_div_mul_sub_tensor")
    h_multi0_tensor = Mul(h_sub_div_mul_sub_tensor,ph_tensor,name="h_multi0_tensor")
    h_multi1_tensor = Mul(h_multi0_tensor,sh_tensor,name="h_multi1_tensor")


    x1_tensor = Sub(x_multi1_tensor,w_multi1_tensor,name="x1_tensor")
    y1_tensor = Sub(y_multi1_tensor,h_multi1_tensor,name="y1_tensor")
    x2_tensor = Add(x_multi1_tensor,w_multi1_tensor,name="x2_tensor")
    y2_tensor = Add(y_multi1_tensor,h_multi1_tensor,name="y2_tensor")
    bosdecoder_output_list = []
    bosdecoder_output_list = [x1_tensor,y1_tensor,x2_tensor,y2_tensor]

    return bosdecoder_output_list

@chalk_common.platform_register
def Reshape(x, new_shape, name=None, bit=16, minimum=None, maximum=None):
    r"""As an entry point into a graph.
    reshape(a, newshape, order='C')
    Gives a new shape to an array without changing its data.

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Reshape(in0,(28,256,2),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Reshape(in0,(28,256,-1),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    sum_shape = 1
    sum_new_shape = 1
    if not isinstance(new_shape, list) and not isinstance(new_shape, tuple):
        raise ValueError('new_shape of Reshape only supports list or tuple')
    for i in range(len(x.shape)):
        sum_shape = sum_shape * x.shape[i]
    new_shape_1 = np.array(new_shape).tolist()
    for j in range(len(new_shape_1)):
        sum_new_shape = sum_new_shape * new_shape_1[j]
    for j in range(len(new_shape_1)):
        if new_shape_1[j] == -1:
            new_shape_1[j] = int((-sum_shape)/sum_new_shape)
            sum_new_shape = (-sum_new_shape)*new_shape_1[j]
    if sum_shape != sum_new_shape:
        raise ValueError('new_shape error!!!')
    tflite.ReshapeOptions.ReshapeOptionsStartNewShapeVector(BUILDER, len(new_shape_1))
    for shape in reversed(new_shape_1):
        BUILDER.PrependInt32(int(shape))
    new_shape_offset = BUILDER.EndVector(len(new_shape_1))
    tflite.ReshapeOptions.ReshapeOptionsStart(BUILDER)
    tflite.ReshapeOptions.ReshapeOptionsAddNewShape(BUILDER, new_shape_offset)
    reshape_options = tflite.ReshapeOptions.ReshapeOptionsEnd(BUILDER)

    input1 = Tensor(data=np.array(new_shape_1).astype('int32'),  bit=32, prefix='new_shape')
    out0 = Tensor(shape=new_shape_1, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Reshape')
    Operator([x, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().RESHAPE,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ReshapeOptions,builtin_options=reshape_options )
    return out0

@chalk_common.platform_register
def Tile(x, multiples, name=None, bit=16, minimum=None, maximum=None):
    r"""Constructs a tensor by tiling a given tensor.

    This operation creates a new tensor by replicating `input` `multiples` times.
    The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
    and the values of `input` are replicated `multiples[i]` times along the 'i'th
    dimension. For example, tiling `[a b c d]` by `[2]` produces
    `[a b c d a b c d]`.

    Args:
    ```text
    input: A `Tensor`. 1-D or higher.
    multiples: A `Tensor`.  Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `input`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Tile(in0,(1,1,2),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```

    """
    if not isinstance(multiples, list) and not isinstance(multiples, tuple):
        raise ValueError('new_shape of Reshape only supports list or tuple')
    multiples_1 = np.array(multiples).tolist()
    if len(x.shape) != len(multiples_1):
        raise ValueError('length shape of input must equal to length shape of multiples!')

    tflite.TileOptions.TileOptionsStart(BUILDER)
    tile_options = tflite.TileOptions.TileOptionsEnd(BUILDER)

    #cal output shape
    output_shape = []
    for i in range(len(x.shape)):
        shape = x.shape[i] * multiples[i]
        output_shape.append(shape)

    input1 = Tensor(data=np.array(multiples).astype('int32'), bit=32, prefix='multiples')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Tile')
    Operator([x, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().TILE,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().TileOptions,builtin_options=tile_options)
    return out0

@chalk_common.platform_register
def Transpose(x, perm, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Transposes x.

    Permutes the dimensions according to `perm`.

    The returned tensor's dimension i will correspond to the input dimension
    `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
    the rank of the input tensor. Hence by default, this operation performs a
    regular matrix transpose on 2-D input Tensors.

    Args:
    ```text
    input: A `Tensor`. 1-D or higher.
    perm: A permutation of the dimensions of `a`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A transposed `Tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Transpose(in0,(0,2,1),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        in1 = sgs_chalk.Input((28,512,4),name='input1')
        out0 = sgs_chalk.Add(in0,in1)
        out1 = sgs_chalk.Transpose(out0,(0,2,1),name='output')
        model = sgs_chalk.Model([in0,in1],out1)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)

    case2:
        in0 = sgs_chalk.Input((28,512,4,1),name='input0')
        out0 = sgs_chalk.Transpose(in0,(0,3,2,1),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if not isinstance(perm, list) and not isinstance(perm, tuple):
        raise ValueError('perm of Transpose only supports list or tuple')
    perm_1 = np.array(perm).tolist()
    if len(x.shape) != len(perm_1):
        raise ValueError('length shape of input must equal to length shape of perm!')

    tflite.TransposeOptions.TransposeOptionsStart(BUILDER)
    transpose_options = tflite.TransposeOptions.TransposeOptionsEnd(BUILDER)

    #cal output shape
    output_shape = []
    for i in range(len(perm_1)):
        output_shape.append(x.shape[perm_1[i]])

    input1 = Tensor(data=np.array(perm).astype('int32'), bit=32, prefix='perm')
    out0 = Tensor(shape=output_shape, name=name, dtype=x.dtype, bit=chalk_common.convert_dtype_to_bit(x.dtype), minimum=minimum, maximum=maximum, prefix='Transpose')
    Operator([x, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().TRANSPOSE,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().TransposeOptions,builtin_options=transpose_options)
    return out0

@chalk_common.platform_register
def Split(x, NumSplits, axis=0, name=None, bit=16, minimum=None, maximum=None):
    r"""
     Splits a tensor into sub tensors.

    NumSplits is an integer, then `value` is split along dimension
    `axis` into NumSplits smaller tensors. This requires that NumSplits evenly
    divides `value.shape[axis]`.

    Args:
    ```text
    input: A `Tensor`. 1-D or higher.
    NumSplits: an integer indicating the number of splits along split_dim
    axis: An integer or scalar `int32` `Tensor`. The dimension along which to
          split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    returns `num_or_size_splits` `Tensor` objects;
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,4,4),name='input0')
        output_list = sgs_chalk.Split(in0,NumSplits=2, axis=2,name=['output0','output1'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)

    case1:
        in0 = sgs_chalk.Input((28,4,3),name='input0')
        output_list = sgs_chalk.Split(in0,NumSplits=2, axis=2,name=['output0','output1'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if x.shape[axis] % NumSplits != 0:
        raise ValueError('Dimension size must be evenly divisible by '+str(NumSplits)+ 'but is '+str(x.shape[axis]))

    axis_value = axis if axis >= 0 else len(x.shape) + axis
    if axis_value >= len(x.shape):
        raise ValueError('axis error!!!')

    tflite.SplitOptions.SplitOptionsStart(BUILDER)
    tflite.SplitOptions.SplitOptionsAddNumSplits(BUILDER, NumSplits)
    Split_options = tflite.SplitOptions.SplitOptionsEnd(BUILDER)

    #cal output shape
    output_shape = list(x.shape)
    output_shape[axis] = x.shape[axis] // NumSplits

    output_tensor_list = []
    for i in range(NumSplits):
        out0 = Tensor(shape=output_shape, name=name[i] if isinstance(name, list) else name,
            bit=bit, minimum=minimum, maximum=maximum, prefix='Split:'+str(i))
        output_tensor_list.append(out0)

    input1 = Tensor(data=np.array([axis_value]).astype('int32'), bit=32, prefix=x.name+'_axis')
    Operator([input1, x], output_tensor_list, tflite.BuiltinOperator.BuiltinOperator().SPLIT,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SplitOptions,builtin_options=Split_options)
    return output_tensor_list

@chalk_common.platform_register
def Split_V(x, SizeSplits, axis=0, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Splits a tensor into sub tensors.

    Sizesplits is a 1-D Tensor (or list), and value is split into len(size_splits)
    elements. The shape of the i-th element has the same size as the `value` except along dimension `axis` where
    the size is size_splits[i].

    Args:
    ```text
    input: A `Tensor`. 1-D or higher.
    SizeSplits: Python list containing the sizes of each output tensor along split_dim.
                the sum of sizes along the split dimension must match that of x
    axis: An integer or scalar `int32` `Tensor`. The dimension along which to
          split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    returns `size_splits` `Tensor` objects;
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,4,4),name='input0')
        output_list = sgs_chalk.Split_V(in0,SizeSplits=[1,2,1], axis=2,name=['output0','output1','output2'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)

    case1:
        in0 = sgs_chalk.Input((28,4,3),name='input0')
        output_list = sgs_chalk.Split_V(in0,NumSplits=[2,2], axis=1,name=['output0','output1'])
        model = sgs_chalk.Model([in0],output_list)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    axis_value = axis if axis >= 0 else len(x.shape) + axis
    if axis_value >= len(x.shape):
        raise ValueError('axis error!!!')

    sum = 0
    for i in range(len(SizeSplits)):
        sum = sum + SizeSplits[i]
    if sum != x.shape[axis]:
        raise ValueError(' the sum of sizes along the split dimension must match that of input ')

    tflite.SplitVOptions.SplitVOptionsStart(BUILDER)
    tflite.SplitVOptions.SplitVOptionsAddNumSplits(BUILDER, len(SizeSplits))
    SplitV_options = tflite.SplitVOptions.SplitVOptionsEnd(BUILDER)

    output_tensor_list = []
    for i in range(len(SizeSplits)):
        #cal output shape
        output_shape = list(x.shape)
        output_shape[axis] = SizeSplits[i]
        out0 = Tensor(shape=output_shape, name=name[i] if isinstance(name, list) else name,
            bit=bit, minimum=minimum, maximum=maximum, prefix='Split:'+str(i))
        output_tensor_list.append(out0)

    input1 = Tensor(data=np.array(SizeSplits).astype('int32'),  bit=32, prefix=x.name+'_sizesplits')
    input2 = Tensor(data=np.array([axis_value]).astype('int32'),  bit=32, prefix=x.name+'_axis')

    Operator([x,input1,input2], output_tensor_list, tflite.BuiltinOperator.BuiltinOperator().SPLIT_V,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SplitVOptions,builtin_options=SplitV_options)
    return output_tensor_list

@chalk_common.platform_register
def Gather(params, indices, axis=0, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Gather slices from `x` axis `axis` according to `indices`.

    `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    Produces an output tensor with shape `params.shape[:axis] + indices.shape +
    params.shape[axis + 1:]`

    Args:
    ```text
    x: A variable `Tensor`. The tensor from which to gather values.
    indices: A `Tensor`.  Must be in range `[0, params.shape[axis])`.
    axis: A `Tensor`. The axis in `params` to gather `indices` from. Defaults to the first
          dimension.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `value`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Gather(in0,[0,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Gather(in0,[[0,1],[0,1]],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Gather(in0,[[0,1],[0,1]],axis=1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case4:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Gather(in0,[1],axis=1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    indices_value = np.array(indices).flatten().tolist()
    for i in range(len(indices_value)):
        if indices_value[i] > params.shape[axis]:
            raise ValueError('indices must be in range `[0, params.shape[axis]) ')

    #cal output shape
    if indices_value[0] == 1 and len(indices_value) == 1:
        output_shape = params.shape
        output_shape = np.array(output_shape).tolist()
        del(output_shape[axis])
    else:
        output_shape = params.shape
        output_shape = np.array(output_shape).tolist()
        if isinstance(indices,int):
            indices = [indices]
        indices_shape = np.array(indices).shape
        indices_shape = np.array(indices_shape).tolist()
        for i in range(len(indices_shape)):
            output_shape.insert(axis+i,indices_shape[i])
        del(output_shape[axis+len(indices_shape)])

    tflite.GatherOptions.GatherOptionsStart(BUILDER)
    tflite.GatherOptions.GatherOptionsAddAxis(BUILDER, axis)
    tflite.GatherOptions.GatherOptionsAddBatchDims(BUILDER,0)
    Gather_options = tflite.GatherOptions.GatherOptionsEnd(BUILDER)

    input1 = Tensor(data=np.array(indices).astype('int32'),  bit=32, prefix='indices')

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Gather')
    Operator([params, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().GATHER,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().GatherOptions, builtin_options=Gather_options)
    return out0

@chalk_common.platform_register
def Average_Pool_2D(x, ksize, stride = (1,1), padding = (0,0,0,0), name=None, bit=16, minimum=None, maximum=None):
    r"""
    Performs the average pooling on the input.
    Each entry in `output` is the mean of the corresponding size `ksize`
    window in `value`.

    Args:
    ```text
    x: A 4-D `Tensor` of shape `[batch, height, width, channels]`
    ksize: A list of `ints` that has length `1`, `2` or `4`. The size of
    the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
    stride of the sliding window for each dimension of the input tensor.
    padding: a tuple (paddingLeft, paddingRight, paddingTop, paddingBottom).
    name: Optional name for the operation.
    ```

    Returns:
    ```text
    The average pooled output tensor.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.Average_Pool_2D(in0,[2,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((1,112,112,3),name='input0')
        out0 = sgs_chalk.Average_Pool_2D(in0,[2,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if len(x.shape) != 4:
        raise ValueError('only support 4dim input !!!')

    if len(padding) != 4:
        raise ValueError('padding only support 4dim input !!!')

    padding_1 = padding
    strideW = stride[1]
    strideH = stride[0]

    paddingLeft = padding_1[0]
    paddingRight = padding_1[1]
    paddingTop = padding_1[2]
    paddingBottom = padding_1[3]

    #SAME = 0  VALID = 1  CAFFE = 2
    padding =2

    filterWidth = ksize[1]
    filterHeight = ksize[0]

    output_shape_H = int((x.shape[1] + 2 * padding_1[0] - ksize[0]) / stride[0]) + 1
    output_shape_W = int((x.shape[2] + 2 * padding_1[1] - ksize[1]) / stride[1]) + 1
    output_shape_C = x.shape[3]
    output_shape = [1, output_shape_H, output_shape_W, output_shape_C]

    tflite.Pool2DOptions.Pool2DOptionsStart(BUILDER)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(BUILDER, padding)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(BUILDER, strideW)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(BUILDER, strideH)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(BUILDER, 0)
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(BUILDER, filterWidth)
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(BUILDER, filterHeight)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(BUILDER, paddingLeft)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(BUILDER, paddingRight)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(BUILDER, paddingTop)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(BUILDER, paddingBottom)
    Average_pool2d_options = tflite.Pool2DOptions.Pool2DOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Average_Pool_2D')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().AVERAGE_POOL_2D,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions,builtin_options=Average_pool2d_options)
    return out0

@chalk_common.platform_register
def Max_Pool_2D(x, ksize, stride = (1,1), padding = (0,0,0,0), name=None, bit=16, minimum=None, maximum=None):
    r"""
    Performs the max pooling on the input.
    Each entry in `output` is the mean of the corresponding size `ksize`
    window in `value`.

    Args:
    ```text
    x: A 4-D `Tensor` of shape `[batch, height, width, channels]`
    ksize: A list of `ints` that has length `1`, `2` or `4`. The size of
    the window for each dimension of the input tensor.
    strides: An int or list of `ints` that has length `1`, `2` or `4`. The
    stride of the sliding window for each dimension of the input tensor.
    padding: a tuple (paddingLeft, paddingRight, paddingTop, paddingBottom).
    name: Optional name for the operation.
    ```

    Returns:
    ```text
    The max pooled output tensor
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.Max_Pool_2D(in0,[2,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((1,112,112,3),name='input0')
        out0 = sgs_chalk.Max_Pool_2D(in0,[2,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if len(x.shape) != 4:
        raise ValueError('only support 4dim input !!!')

    if len(padding) != 4:
        raise ValueError('padding only support 4dim input !!!')

    padding_1 = padding
    strideW = stride[1]
    strideH = stride[0]

    paddingLeft = padding_1[0]
    paddingRight = padding_1[1]
    paddingTop = padding_1[2]
    paddingBottom = padding_1[3]

    #SAME = 0  VALID = 1  CAFFE = 2
    padding =2

    filterWidth = ksize[1]
    filterHeight = ksize[0]

    output_shape_H = int((x.shape[1] + 2 * padding_1[0] - ksize[0]) / stride[0]) + 1
    output_shape_W = int((x.shape[2] + 2 * padding_1[1] - ksize[1]) / stride[1]) + 1
    output_shape_C = x.shape[3]
    output_shape = [1, output_shape_H, output_shape_W, output_shape_C]

    tflite.Pool2DOptions.Pool2DOptionsStart(BUILDER)
    tflite.Pool2DOptions.Pool2DOptionsAddPadding(BUILDER, padding)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideW(BUILDER, strideW)
    tflite.Pool2DOptions.Pool2DOptionsAddStrideH(BUILDER, strideH)
    tflite.Pool2DOptions.Pool2DOptionsAddFusedActivationFunction(BUILDER, 0)
    tflite.Pool2DOptions.Pool2DOptionsAddFilterWidth(BUILDER, filterWidth)
    tflite.Pool2DOptions.Pool2DOptionsAddFilterHeight(BUILDER, filterHeight)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingLeft(BUILDER, paddingLeft)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingRight(BUILDER, paddingRight)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingTop(BUILDER, paddingTop)
    tflite.Pool2DOptions.Pool2DOptionsAddPaddingBottom(BUILDER, paddingBottom)
    Max_pool2d_options = tflite.Pool2DOptions.Pool2DOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Max_Pool_2D')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().MAX_POOL_2D,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().Pool2DOptions,builtin_options=Max_pool2d_options)
    return out0

@chalk_common.platform_register
def Conv2d(x, weight, bias = None,  Activation = 'NONE',stride = (1,1), padding = (0,0,0,0), dilation = (1,1), name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    x : input tensor of shape(1, iH, iW,in_channels)
    weight: filters of shape(out_channels, kH, kW, in_channels)
    bias: optional bias tensor of shape(out_channels).Default:None
    stride: the stride of the convolving kernel. Can be a single number or a tuple(sH,sW).Default: 1
    padding: a tuple (paddingLeft, paddingRight, paddingTop, paddingBottom).
    dilation: the spacing between kernel elements. Can be a single number or a tuple(dH,dW).Default: 1
    Activation: only support NONE/RELU/RELU_N1_TO_1/RELU6
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        in1 = sgs_chalk.Input((8,3,3,4),name='input1')
        out0 = sgs_chalk.Conv2d(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((8,3,3,4), dtype=np.float32)
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.Conv2d(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((8,3,3,4), dtype=np.float32)
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.Conv2d(in0,input1,bias = None, Activation = 'RELU',name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if len(x.shape) != 4:
        raise ValueError('only support 4dim input !!!')
    if len(weight.shape) != 4:
        raise ValueError('only support 4dim weight !!!')
    if x.shape[3] != weight.shape[3]:
        raise ValueError('input in channels and weight in channnls must be same!!!')
    if len(padding) != 4:
        raise ValueError('padding only support 4dim input !!!')

    if isinstance(weight, np.ndarray):
        if weight.astype != 'float32':
            weight = weight.astype('float32')
        weight = Tensor(data=weight,name='weight')
    elif isinstance(weight, Tensor):
        weight = weight
    else:
        raise ValueError('Not support this type:', type(weight))

    if bias == None:
        bias = np.zeros(weight.shape[0], dtype=np.float32)
        bias = Tensor(data=bias, name='bias', bit=bit, prefix='bias')
    if isinstance(bias, np.ndarray):
        bias = Tensor(data=bias,name='bias')
    elif isinstance(bias, Tensor):
        bias = bias
    else:
        raise ValueError('Not support this type:', type(bias))



    padding_1 = padding
    kernel_size = [weight.shape[1],weight.shape[2]]
    strideW = stride[1]
    strideH = stride[0]
    dilationWFactor = dilation[1]
    dilationHFactor = dilation[0]
    paddingLeft = padding_1[0]
    paddingRight = padding_1[1]
    paddingTop = padding_1[2]
    paddingBottom = padding_1[3]

    #SAME = 0  VALID = 1  CAFFE = 2
    padding =2

    if Activation == 'NONE':
        fusedActivationFunction = 0
    elif Activation == 'RELU':
        fusedActivationFunction = 1
    elif Activation == 'RELU_N1_TO_1':
        fusedActivationFunction = 2
    elif Activation == 'RELU6':
        fusedActivationFunction = 3
    else:
        raise ValueError('Not support this activation !!!')

    output_shape_H = int((x.shape[1] + 2 * padding_1[0] - dilation[0] * (kernel_size[0]-1) -1) / stride[0]) + 1
    output_shape_W = int((x.shape[2] + 2 * padding_1[1] - dilation[1] * (kernel_size[1]-1) -1) / stride[1]) + 1
    output_shape_C = weight.shape[0]
    output_shape = [1, output_shape_H, output_shape_W, output_shape_C]

    tflite.Conv2DOptions.Conv2DOptionsStart(BUILDER)
    tflite.Conv2DOptions.Conv2DOptionsAddPadding(BUILDER, padding)
    tflite.Conv2DOptions.Conv2DOptionsAddStrideW(BUILDER, strideW)
    tflite.Conv2DOptions.Conv2DOptionsAddStrideH(BUILDER, strideH)
    tflite.Conv2DOptions.Conv2DOptionsAddFusedActivationFunction(BUILDER, fusedActivationFunction)
    tflite.Conv2DOptions.Conv2DOptionsAddDilationWFactor(BUILDER, dilationWFactor)
    tflite.Conv2DOptions.Conv2DOptionsAddDilationHFactor(BUILDER, dilationHFactor)
    tflite.Conv2DOptions.Conv2DOptionsAddPaddingLeft(BUILDER, paddingLeft)
    tflite.Conv2DOptions.Conv2DOptionsAddPaddingRight(BUILDER, paddingRight)
    tflite.Conv2DOptions.Conv2DOptionsAddPaddingTop(BUILDER, paddingTop)
    tflite.Conv2DOptions.Conv2DOptionsAddPaddingBottom(BUILDER, paddingBottom)
    Conv2D_options = tflite.Conv2DOptions.Conv2DOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Conv2d')
    Operator([x,weight,bias], [out0], tflite.BuiltinOperator.BuiltinOperator().CONV_2D,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().Conv2DOptions,builtin_options=Conv2D_options)
    return out0

@chalk_common.platform_register
def DepthWiseConv2D(x, weight, bias = None, Activation = 'NONE', stride = (1,1), padding = (0,0,0,0), dilation = (1,1), name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    x : input tensor of shape(1, iH, iW,in_channels)
    weight: filters of shape(1, kH, kW, out_channels) in_channels = out_channels
    bias: optional bias tensor of shape(out_channels).Default:None
    stride: the stride of the convolving kernel. Can be a single number or a tuple(sH,sW).Default: 1
    padding: a tuple (paddingLeft, paddingRight, paddingTop, paddingBottom).
    dilation: the spacing between kernel elements. Can be a single number or a tuple(dH,dW).Default: 1
    Activation: only support NONE/RELU/RELU_N1_TO_1/RELU6
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        in1 = sgs_chalk.Input((1,3,3,4),name='input1')
        out0 = sgs_chalk.DepthWiseConv2D(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((1,3,3,4), dtype=np.float32)
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.DepthWiseConv2D(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((1,3,3,4), dtype=np.float32)
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        out0 = sgs_chalk.DepthWiseConv2D(in0,input1,bias = None, Activation = 'RELU',name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if len(x.shape) != 4:
        raise ValueError('only support 4dim input !!!')


    if len(padding) != 4:
        raise ValueError('padding only support 4dim input !!!')
    if isinstance(weight, np.ndarray):
        weight = Tensor(data=weight,name='weight')
    elif isinstance(weight, Tensor):
        weight = weight
    else:
        raise ValueError('Not support this type:', type(weight))

    if len(weight.shape) != 4:
        raise ValueError('only support 4dim weight !!!')
    if weight.shape[0] != 1:
        raise ValueError('depthwise weight out_channels must be 1 !!!')
    if x.shape[3] != weight.shape[3]:
        raise ValueError('input in_channels and weight out_channels must be same!!!')

    if bias is None:
        bias = np.zeros(weight.shape[3], dtype=np.float32)
        bias = Tensor(data=bias, name='bias', bit=bit, prefix='bias')
    elif isinstance(bias, np.ndarray):
        bias = Tensor(data=bias,name='bias')
    elif isinstance(bias, Tensor):
        bias = bias
    else:
        raise ValueError('Not support this type:', type(weight))

    if weight.is_const == False:
        if weight.shape[1] != 3 and weight.shape[2] != 3:
            if weight.shape[1] != 6 and weight.shape[2] != 6:
                if weight.shape[1] != 9 and weight.shape[2] != 9 :
                    raise ValueError('DepthwiseConv2d dynamic weight only support 3*3 or 6*6 or 9*9!')

    padding_1 = padding
    kernel_size = [weight.shape[1],weight.shape[2]]
    strideW = stride[1]
    strideH = stride[0]
    dilationWFactor = dilation[1]
    dilationHFactor = dilation[0]
    paddingLeft = padding_1[0]
    paddingRight = padding_1[1]
    paddingTop = padding_1[2]
    paddingBottom = padding_1[3]

    #SAME = 0  VALID = 1  CAFFE = 2
    padding =2

    if Activation == 'NONE':
        fusedActivationFunction = 0
    elif Activation == 'RELU':
        fusedActivationFunction = 1
    elif Activation == 'RELU_N1_TO_1':
        fusedActivationFunction = 2
    elif Activation == 'RELU6':
        fusedActivationFunction = 3
    else:
        raise ValueError('Not support this activation !!!')

    output_shape_H = int((x.shape[1] + 2 * padding_1[0] - dilation[0] * (kernel_size[0]-1) -1) / stride[0]) + 1
    output_shape_W = int((x.shape[2] + 2 * padding_1[1] - dilation[1] * (kernel_size[1]-1) -1) / stride[1]) + 1
    output_shape_C = x.shape[3]
    output_shape = [1, output_shape_H, output_shape_W, output_shape_C]

    depthMultiplier = 0

    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsStart(BUILDER)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPadding(BUILDER, padding)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideW(BUILDER, strideW)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddStrideH(BUILDER, strideH)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(BUILDER, depthMultiplier)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddFusedActivationFunction(BUILDER, fusedActivationFunction)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationWFactor(BUILDER, dilationWFactor)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDilationHFactor(BUILDER, dilationHFactor)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingLeft(BUILDER, paddingLeft)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingRight(BUILDER, paddingRight)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingTop(BUILDER, paddingTop)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddPaddingBottom(BUILDER, paddingBottom)
    tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsAddDepthMultiplier(BUILDER,depthMultiplier)
    DepthWiseConv2D_options = tflite.DepthwiseConv2DOptions.DepthwiseConv2DOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='DepthwiseConv2d')
    Operator([x,weight,bias], [out0], tflite.BuiltinOperator.BuiltinOperator().DEPTHWISE_CONV_2D,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().DepthwiseConv2DOptions,builtin_options=DepthWiseConv2D_options)
    return out0

@chalk_common.platform_register
def BatchMatMul(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""

    Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

    The inputs must  be tensors of rank >= 2
    where the inner 2 dimensions specify valid matrix multiplication arguments,
    and any further outer dimensions match.

    Args:
    ```text
    x: input variable tensor and rank > 1.
    y: input tensor with same type and rank as x,can be variable or const tensor.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),name='input0')
        in1 = sgs_chalk.Input((1,28,28,4),name='input1')
        out0 = sgs_chalk.BatchMatMul(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.ones((1,28,28,4), dtype=np.float32)
        in0 = sgs_chalk.Input((1,28,512,4),name='input0')
        out0 = sgs_chalk.BatchMatMul(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        in1 = Tensor(data=y)
    elif isinstance(y, Tensor):
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if len(x.shape) != len(y.shape):
        raise ValueError('input tensor shape must be same!!!')
    for i in range(len(x.shape)-2):
        if x.shape[i] != y.shape[i]:
            raise ValueError('input tensor shape error!!!')
    if x.shape[-1] != y.shape[-1]:
        raise ValueError('input tensor inner shape must be same!!!')

    #cal output shape
    output_shape = list(x.shape)
    output_shape.pop()
    output_shape.pop()
    output_shape.append(x.shape[-2])
    output_shape.append(y.shape[-2])

    #creat option
    asymmetricQuantizeInputs = 0
    changeA = False
    changeB = False
    tflite.BatchMatMulOptions.BatchMatMulOptionsStart(BUILDER)
    tflite.BatchMatMulOptions.BatchMatMulOptionsAddAdjX(BUILDER, changeA)
    tflite.BatchMatMulOptions.BatchMatMulOptionsAddAdjY(BUILDER, changeB)
    tflite.BatchMatMulOptions.BatchMatMulOptionsAddAsymmetricQuantizeInputs(BUILDER, asymmetricQuantizeInputs)
    BatchMatMul_options = tflite.BatchMatMulOptions.BatchMatMulOptionsEnd(BUILDER)

    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='BatchMatMul')

    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().BATCH_MATMUL,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().BatchMatMulOptions,builtin_options=BatchMatMul_options)
    return out0

@chalk_common.platform_register
def Pad(x, paddings, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Pads a tensor.

    This operation pads a `tensor` according to the `paddings` you specify.
    `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
    `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
    many values to add before the contents of `tensor` in that dimension, and
    `paddings[D, 1]` indicates how many values to add after the contents of
    `tensor` in that dimension.

    The padded size of each dimension D of the output is:
    `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

    Args:
    ```text
    tensor: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Pad(in0,[[0,0],[10,10]],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512,4),name='input0')
        out0 = sgs_chalk.Pad(in0,[[0,0],[10,10],[1,1]],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512,512,4),name='input0')
        out0 = sgs_chalk.Pad(in0,[[1,1],[2,2],[3,3],[4,4]],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """

    tflite.PadOptions.PadOptionsStart(BUILDER)
    pad_options = tflite.PadOptions.PadOptionsEnd(BUILDER)

    #cal output shape
    output_shape = [ ]
    input_shape = np.array(x.shape).tolist()
    for i in range(len(input_shape)):
        outputshape_1 = input_shape[i] + paddings[i][0] + paddings[i][1]
        output_shape.append(outputshape_1)

    input1 = Tensor(data=np.array(paddings).astype('int32'),  bit=32, prefix='paddings')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Pad')
    Operator([x, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().PAD,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().PadOptions,builtin_options=pad_options)
    return out0

@chalk_common.platform_register
def Softmax(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes softmax activations.

    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Softmax(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """

    tflite.SoftmaxOptions.SoftmaxOptionsStart(BUILDER)
    tflite.SoftmaxOptions.SoftmaxOptionsAddBeta(BUILDER, 0.0)
    Softmax_options = tflite.SoftmaxOptions.SoftmaxOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Softmax')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().SOFTMAX,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SoftmaxOptions,builtin_options=Softmax_options)
    return out0

@chalk_common.platform_register
def Logistic(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    `output = 1 / (1 + exp(-x))`

    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Logistic(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Logistic')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().LOGISTIC)
    return out0

@chalk_common.platform_register
def Tanh(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Tanh(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Tanh')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().TANH)
    return out0

@chalk_common.platform_register
def Relu(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes rectified linear: `max(x, 0)`.

    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Relu(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Relu')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().RELU)
    return out0

@chalk_common.platform_register
def Relu6(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes Rectified Linear 6: `min(max(features, 0), 6)

    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Relu6(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Relu6')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().RELU6)
    return out0

@chalk_common.platform_register
def Relu_N1_TO_1(x, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    tensor: A `Tensor`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Relu_N1_TO_1(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Relu_N1_TO_1')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().RELU_N1_TO_1)
    return out0

@chalk_common.platform_register
def LeakyRelu(x, alpha=0.0, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Compute the Leaky ReLU activation function.

    Args:
    ```text
    tensor: A `Tensor`.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.LeakyRelu(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    tflite.LeakyReluOptions.LeakyReluOptionsStart(BUILDER)
    tflite.LeakyReluOptions.LeakyReluOptionsAddAlpha(BUILDER, alpha)
    LeakyRelu_options = tflite.LeakyReluOptions.LeakyReluOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='LeakyRelu')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().LEAKY_RELU,
             builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().LeakyReluOptions,builtin_options=LeakyRelu_options)
    return out0

@chalk_common.platform_register
def Prelu(x, slope, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Parametric Rectified Linear Unit.

    It follows:
    f(x) = alpha * x for x < 0,
    f(x) = x for x >= 0,
    where `alpha` is a learned array with the same shape as x.


    Args:
    ```text
    tensor: A `Tensor`.
    slope: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. Has the same type as `tensor`.
    ```

    Examples:
    ```python
    case0:
        input1 = np.zeros((512,), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Prelu(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Prelu(in0,5,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((512,), dtype=np.float32)
        in0 = sgs_chalk.Input((3,28,512),name='input0')
        out0 = sgs_chalk.Prelu(in0,5,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(slope, np.ndarray):
        if len(slope.shape) != len(x.shape)-1 and len(slope.shape) != 1:
            raise ValueError('if slope not value, len(slope.shape) must equal len(x.shape)-1  or 1')

        slope_shape = np.array(slope.shape).tolist()

        if len(slope.shape) == len(x.shape)-1:
            for i in range(len(slope_shape)):
                if slope_shape[i] != x.shape[i+1] and slope_shape[i] != 1:
                    raise ValueError('slope_shape[i] must be equal to x.shape[i] or 1')
        else:
            if slope_shape[0] != x.shape[-1]:
                raise ValueError('if len(slope_shape)=1,slope shape must equal input shape[-1]')
        slope = slope
    else:
        slope = [slope]

    input1 = Tensor(data=np.array(slope).astype('float32'), bit=16, prefix='slope')
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Prelu')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().PRELU)
    return out0

@chalk_common.platform_register
def Mean(x, axis, keep_dim=1, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes the mean of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.

    Args:
    ```text
    tensor: A `Tensor`.
    axis: The dimensions to reduce.
    keep_dim:whether keep dim or not
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. The reduced tensor.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.Mean(in0,[1,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if len(x.shape) != 4:
        raise ValueError('input shape dims must be 4')
    if not isinstance(axis, list) and not isinstance(axis, tuple):
        raise ValueError('axis of Mean only supports list or tuple')
    if axis[0] != 1 and axis[1] != 2:
        raise ValueError('axis only support [1,2]')

    #cal output shape
    output_shape = []
    output_shape = [x.shape[0],1,1,x.shape[3]]

    tflite.ReducerOptions.ReducerOptionsStart(BUILDER)
    tflite.ReducerOptions.ReducerOptionsAddKeepDims(BUILDER,keep_dim)
    mean_options = tflite.ReducerOptions.ReducerOptionsEnd(BUILDER)

    input1 = Tensor(data=np.array(axis).astype('int32'), bit=32, prefix='axis')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Mean')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().MEAN,builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ReducerOptions,builtin_options=mean_options)
    return out0

@chalk_common.platform_register
def Sum(x, axis, keep_dim=1, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes the sum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.

    Args:
    ```text
    tensor: A `Tensor`.
    axis: The dimensions to reduce.
    keep_dim:whether keep dim or not
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. The reduced tensor.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.Sum(in0,[0,1,2,3],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.Sum(in0,[1,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """

    if not isinstance(axis, list) and not isinstance(axis, tuple):
        raise ValueError('axis of Mean only supports list or tuple')

    if len(axis) > len(x.shape):
        raise ValueError('axis length cannot be greater than input length')

    #cal output shape
    output_shape = list(x.shape)
    for i in range(len(axis)):
        output_shape[axis[i]] = 1

    tflite.ReducerOptions.ReducerOptionsStart(BUILDER)
    tflite.ReducerOptions.ReducerOptionsAddKeepDims(BUILDER,keep_dim)
    sum_options = tflite.ReducerOptions.ReducerOptionsEnd(BUILDER)

    input1 = Tensor(data=np.array(axis).astype('int32'), bit=32, prefix='axis')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Sum')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().SUM,builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ReducerOptions,builtin_options=sum_options)
    return out0

@chalk_common.platform_register
def Reduce_Max(x, axis, keep_dim=1, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Computes the largest element across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `axis`.

    Args:
    ```text
    tensor: A `Tensor`.
    axis: The dimensions to reduce.
    keep_dim:whether keep dim or not
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor`. The reduced tensor.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.Reduce_Max(in0,[0,1,2,3],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case0:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.Reduce_Max(in0,[1,2],name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """

    if not isinstance(axis, list) and not isinstance(axis, tuple):
        raise ValueError('axis of Mean only supports list or tuple')

    if len(axis) > len(x.shape):
        raise ValueError('axis length cannot be greater than input length')

    #cal output shape
    output_shape = list(x.shape)
    for i in range(len(axis)):
        output_shape[axis[i]] = 1

    tflite.ReducerOptions.ReducerOptionsStart(BUILDER)
    tflite.ReducerOptions.ReducerOptionsAddKeepDims(BUILDER,keep_dim)
    reducemax_options = tflite.ReducerOptions.ReducerOptionsEnd(BUILDER)

    input1 = Tensor(data=np.array(axis).astype('int32'), bit=32, prefix='axis')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Reduce_Max')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().REDUCE_MAX,builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ReducerOptions,builtin_options=reducemax_options)
    return out0

@chalk_common.platform_register
def Greater(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""
    greater(x, y, name=None)
    Returns the truth value of (x > y) element-wise.

    ```text
    Args:
    x: A `Tensor`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor` of type `bool`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Greater(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Greater(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Greater(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))
    tflite.GreaterOptions.GreaterOptionsStart(BUILDER)
    greater_options = tflite.GreaterOptions.GreaterOptionsEnd(BUILDER)
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Greater')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().GREATER,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().GreaterOptions,builtin_options=greater_options)
    return out0

@chalk_common.platform_register
def Less(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""
    greater(x, y, name=None)
    Returns the truth value of (x < y) element-wise.

    Args:
    ```text
    x: A `Tensor`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).
    ```

    Returns:
    ```text
    A `Tensor` of type `bool`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Less(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Less(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Less(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))
    tflite.LessOptions.LessOptionsStart(BUILDER)
    less_options = tflite.LessOptions.LessOptionsEnd(BUILDER)
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Less')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().LESS,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().LessOptions,builtin_options=less_options)
    return out0

@chalk_common.platform_register
def Equal(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns the truth value of (x == y) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Equal(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Equal(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Equal(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.EqualOptions.EqualOptionsStart(BUILDER)
    equal_options = tflite.EqualOptions.EqualOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Equal')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().EQUAL,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().EqualOptions,builtin_options=equal_options)
    return out0

@chalk_common.platform_register
def NotEqual(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns the truth value of (x != y) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.NotEqual(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.NotEqual(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.NotEqual(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.NotEqualOptions.NotEqualOptionsStart(BUILDER)
    notequal_options = tflite.NotEqualOptions.NotEqualOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='NotEqual')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().NOT_EQUAL,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().NotEqualOptions,builtin_options=notequal_options)
    return out0

@chalk_common.platform_register
def GreaterEqual(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns the truth value of (x >= y) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.GreaterEqual(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.GreaterEqual(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.GreaterEqual(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.GreaterEqualOptions.GreaterEqualOptionsStart(BUILDER)
    greaterequal_options = tflite.GreaterEqualOptions.GreaterEqualOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='GreaterEqual')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().GREATER_EQUAL,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().GreaterEqualOptions,builtin_options=greaterequal_options)
    return out0

@chalk_common.platform_register
def LogicalAnd(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns the truth value of x AND y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).sss
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.LogicalAnd(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.LogicalAnd(in0,1.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.LogicalAnd(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.LogicalAndOptions.LogicalAndOptionsStart(BUILDER)
    logicaland_options = tflite.LogicalAndOptions.LogicalAndOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='LogicalAnd')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().LOGICAL_AND,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().LogicalAndOptions,builtin_options=logicaland_options)
    return out0

@chalk_common.platform_register
def LogicalNot(x, y, name=None, bit=16, minimum=None, maximum=None):
    r""" Returns the truth value of NOT x element-wise.

    Args:
    ```test
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```test
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.LogicalNot(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.LogicalNot(in0,1.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.LogicalNot(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.LogicalNotOptions.LogicalNotOptionsStart(BUILDER)
    logicalnot_options = tflite.LogicalNotOptions.LogicalNotOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='LogicalNot')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().LOGICAL_NOT,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().LogicalNotOptions,builtin_options=logicalnot_options)
    return out0

@chalk_common.platform_register
def Maximum(x, y, name=None, bit=16, minimum=None, maximum=None):
    r""" Returns the max of x and y (i.e. x > y ? x : y) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Maximum(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Maximum(in0,1.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Maximum(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.MaximumMinimumOptions.MaximumMinimumOptionsStart(BUILDER)
    maximumminimum_options = tflite.MaximumMinimumOptions.MaximumMinimumOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Maximum')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().MAXIMUM,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().MaximumMinimumOptions,builtin_options=maximumminimum_options)
    return out0

@chalk_common.platform_register
def Minimum(x, y, name=None, bit=16, minimum=None, maximum=None):
    r""" Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.Tensor` of type `bool
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Minimum(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Minimum(in0,1.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Minimum(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.MaximumMinimumOptions.MaximumMinimumOptionsStart(BUILDER)
    maximumminimum_options = tflite.MaximumMinimumOptions.MaximumMinimumOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Minimum')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().MINIMUM,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().MaximumMinimumOptions,builtin_options=maximumminimum_options)
    return out0


@chalk_common.platform_register
def Exp(x, name=None, bit=16, minimum=None, maximum=None):
    r""" Computes exponential of x element-wise. (y = e^x).


    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Exp(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    tflite.ExpOptions.ExpOptionsStart(BUILDER)
    exp_options = tflite.ExpOptions.ExpOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Exp')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().EXP,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ExpOptions,builtin_options=exp_options)
    return out0

@chalk_common.platform_register
def Select(condition, x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""
    where(condition, x=None, y=None, name=None)

    Return the elements, either from `x` or `y`, depending on the `condition`. (deprecated)
    If both `x` and `y` are None, then this operation returns the coordinates of
    true elements of `condition`.  The coordinates are returned in a 2-D tensor
    where the first dimension (rows) represents the number of true elements, and
    the second dimension (columns) represents the coordinates of the true
    elements. Keep in mind, the shape of the output tensor can vary depending on
    how many true values there are in input. Indices are output in row-major
    order.

    If both non-None, `x` and `y` must have the same shape.
    The `condition` tensor must be a scalar if `x` and `y` are scalar.
    If `x` and `y` are vectors of higher rank, then `condition` must be either a
    vector with size matching the first dimension of `x`, or must have the same
    shape as `x`.

    The `condition` tensor acts as a mask that chooses, based on the value at each
    element, whether the corresponding element / row in the output should be taken
    from `x` (if true) or `y` (if false).

    If `condition` is a vector and `x` and `y` are higher rank matrices, then it
    chooses which row (outer dimension) to copy from `x` and `y`. If `condition`
    has the same shape as `x` and `y`, then it chooses which element to copy from
    `x` and `y`.


    Args:
    ```text
    condition: A `Tensor` of type `float32`
    x: A Tensor which may have the same shape as `condition`. If `condition` is
    rank 1, `x` may have higher rank, but its first dimension must match the
    size of `condition`.
    y: A `tensor` with the same shape and type as `x`.
    name: A name of the operation (optional)
    ```

    Returns:
    ```text
    A `Tensor` with the same type and shape as `x`, `y` if they are non-None.
    Otherwise, a `Tensor` with shape `(num_true, rank(condition))`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        in2 = sgs_chalk.Input((28,512),name='input2')
        out0 = sgs_chalk.Select(in0,in1,in2,name='output')
        model = sgs_chalk.Model([in0,in1,in2],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Select(in0,5.0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case2:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Select(input1,in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')
    ```
    """

    #if ndarry change into tensor
    if isinstance(condition, np.ndarray):
        condition = Tensor(data=condition)
    elif isinstance(condition, float):
        condition = Tensor(data=np.array([condition]).astype(np.float32))
    elif isinstance(condition, Tensor):
        condition = condition
    else:
        raise ValueError('Not support this type:', type(condition))

    if isinstance(x, np.ndarray):
        x = Tensor(data=x)
    elif isinstance(x, float):
        x = Tensor(data=np.array([x]).astype(np.float32))
    elif isinstance(x, Tensor):
        x = x
    else:
        raise ValueError('Not support this type:', type(x))

    if isinstance(y, np.ndarray):
        y = Tensor(data=y)
    elif isinstance(y, float):
        y = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        y = y
    else:
        raise ValueError('Not support this type:', type(y))

    #cal output_shape

    cal_max = lambda x, y, z: (x if (x > y) else y) if ((x if (x > y) else y) > z) else z

    len_output = cal_max(len(condition.shape),len(x.shape),len(y.shape))

    #expand dims
    if len(condition.shape) == 1:
        for i in range(len_output-1):
            condition_data = np.expand_dims(condition.np_data, axis=0)
        condition_shape = condition_data.shape
    else:
        condition_shape = y.shape

    if len(x.shape) == 1:
        for i in range(len_output-1):
            x_data = np.expand_dims(x.np_data, axis=0)
        x_shape = x_data.shape
    else:
        x_shape = x.shape

    if len(y.shape) == 1:
        for i in range(len_output-1):
            y_data = np.expand_dims(y.np_data, axis=0)
        y_shape = y_data.shape

    else:
        y_shape = y.shape

    output_shape = []
    for i in range(len_output):
        if condition_shape[i] == x_shape[i] and x_shape[i] == y_shape[i]:
            value = condition_shape[i]
        else:
            value = cal_max(condition_shape[i],x_shape[i],y_shape[i])
        output_shape.append(value)
    #whether broadcast or not
    def is_need_broadcast(len_dims, tensor_dims, output_shape):
        tmp_list = []
        tile_param = []
        if len_dims == 4:
            if (tensor_dims == [1, 1, 1, 1] or tensor_dims == [1, output_shape[1], 1, 1]
                    or tensor_dims == output_shape):
                return False, None
            else:
                for index, i in enumerate(output_shape):
                    if i != tensor_dims[index]:
                        tile_param.append(output_shape[index])
                    else:
                        tile_param.append(1)
                return True, tile_param
        elif len_dims == 1:
            return False, None
        else:
            for x in range(len(tensor_dims)-1):
                tmp_list.append(1)
            tmp_list.append(output_shape[-1])
            if tensor_dims == tmp_list or tensor_dims == output_shape:
                return False, None
            else:
                for index, i in enumerate(output_shape):
                    if i != tensor_dims[index]:
                        tile_param.append(output_shape[index])
                    else:
                        tile_param.append(1)
                return True, tile_param
    #whether const
    if condition.is_const == True:
        where_const_tensor = condition.np_data
        is_need, tile_param = is_need_broadcast(len(condition.shape),
                                                        condition.shape, output_shape)
        if is_need:
            where_const_ndarray = np.reshape(where_const_tensor, condition.shape)
            where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
            x0 = Tensor(data=np.array(where_const_tensor_1).astype(np.float32))
        else:
            x0 = condition
    else:
        x0 = condition
    if x.is_const == True:
        where_const_tensor = x.np_data
        is_need, tile_param = is_need_broadcast(len(x.shape),
                                                        x.shape, output_shape)
        if is_need:
            where_const_ndarray = np.reshape(where_const_tensor, x.shape)
            where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
            x1 = Tensor(data=np.array(where_const_tensor_1).astype(np.float32))
        else:
            x1 = x
    else:
        x1 = x
    if y.is_const == True:
        where_const_tensor = y.np_data
        is_need, tile_param = is_need_broadcast(len(y.shape),
                                                        y.shape, output_shape)
        if is_need:
            where_const_ndarray = np.reshape(where_const_tensor, y.shape)
            where_const_tensor_1 = np.tile(where_const_ndarray, tile_param)
            x2 = Tensor(data=np.array(where_const_tensor_1).astype(np.float32))
        else:
            x2 = y
    else:
        x2 = y


    tflite.SelectOptions.SelectOptionsStart(BUILDER)
    select_options = tflite.SelectOptions.SelectOptionsEnd(BUILDER)
    # input1 = np.zeros((x.shape), dtype=np.float32)
    # input1 = Tensor(data=input1,  bit=33, prefix='where_condition')
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Select')
    Operator([x0, x1, x2], [out0], tflite.BuiltinOperator.BuiltinOperator().SELECT,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().SelectOptions,builtin_options=select_options)
    return out0


@chalk_common.platform_register
def Input(shape, name=None, dtype='float32', minimum=None, maximum=None):
    r"""As an entry point into a graph.

    Args:
    ```test
    shape: `Tuple`.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```test
    A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')

    case1:
        in0 = sgs_chalk.Input((28,512),datatype='complex64', name='input0')
    ```
    """
    return Tensor(shape=shape, name=name, dtype=dtype,
        bit=chalk_common.convert_dtype_to_bit(dtype), minimum=minimum, maximum=maximum, prefix='Input')

@chalk_common.platform_register
def Fix2Float(x,name=None,bit=33,minimum=None,maximum=None):
    r"""Convert fix to float.

    Args:
    ```test
    x: A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```test
    A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Fix2Float(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Fix2Float')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'Fix2Float',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def Float2Fix(x,name=None,bit=16,minimum=None,maximum=None):
    r"""Convert float to fix.

    Args:
    ```test
    x: A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```test
    A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Float2Fix(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Float2Fix')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'Float2Fix',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def FFT(x,name=None):
    r"""FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms.  The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

    Args:
    ```test
    x : A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```test
    out : A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.FFT(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    def isPower(n):
        if n <= 1:
            return False
        i = 1
        while i <= n:
            if i == n:
                return True
            i <<= 1
        return False
    #if len(x.shape) != 2:
    #    raise ValueError('FFT only support 2 dims')
    if not isPower(x.shape[-1]):
        raise ValueError('FFT only support input number of 2^n(n!=0)')
    if x.shape[-1] > 8192:
        raise ValueError('FFT only support input number less than 8192')
    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    if x.shape[-1] >= 4:
        input1 = np.zeros((x.shape[-1]//4), dtype=np.complex64)
    else:
        input1 = np.zeros(int(x.shape[-1]), dtype=np.complex64)
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='FFT')
    input1 = Tensor(data=input1, name=name, bit=65, prefix='FFT_input1')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'FFT',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def IFFT(x,name=None):
    r"""Compute the  inverse discrete Fourier Transform.
    This function computes the inverse of the
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.

    Args:
    ```test
    x : A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```test
    out : A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.IFFT(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    def isPower(n):
        if n <= 1:
            return False
        i = 1
        while i <= n:
            if i == n:
                return True
            i <<= 1
        return False
    #if len(x.shape) != 2:
    #    raise ValueError('FFT only support 2 dims')
    if not isPower(x.shape[-1]):
        raise ValueError('IFFT only support input number of 2^n(n!=0)')
    if x.shape[-1] > 8192:
        raise ValueError('IFFT only support input number less than 8192')
    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    if x.shape[-1] >= 4:
        input1 = np.zeros((x.shape[-1]//4), dtype=np.complex64)
    else:
        input1 = np.zeros(int(x.shape[-1]), dtype=np.complex64)
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65,  prefix='IFFT')
    input1 = Tensor(data=input1, name=name, bit=65,  prefix='IFFT_input1')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'IFFT',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def RFFT(x,name=None):
    r""" Compute discrete Fourier Transform for real input.
    This function computes the  discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).


    Args:
    ```text
    x : A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    out : A Variable `Tensor`. output shape is (x.shape[0], int((x.shape[1]/2)+1))
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.RFFT(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    def isPower(n):
        if n <= 1:
            return False
        i = 1
        while i <= n:
            if i == n:
                return True
            i <<= 1
        return False
    #if len(x.shape) != 2:
    #    raise ValueError('FFT only support 2 dims')
    if not isPower(x.shape[-1]):
        raise ValueError('RFFT only support input number of 2^n(n!=0)')
    if x.shape[-1] > 8192:
        raise ValueError('RFFT only support input number less than 8192')
    if x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')
    if x.shape[-1] >= 4:
        input1 = np.zeros((x.shape[-1]//4), dtype=np.complex64)
    else:
        input1 = np.zeros(int(x.shape[-1]), dtype=np.complex64)

    #cal output shape
    output_shape = [ ]
    input_shape = np.array(x.shape).tolist()
    for i in range(len(input_shape)):
        outputshape_1 = input_shape[i]
        output_shape.append(outputshape_1)
    output_shape[-1] = input_shape[-1]//2+1

    out0 = Tensor(shape=output_shape, dtype='complex64', name=name, bit=65,  prefix='RFFT')
    input1 = Tensor(data=input1, name=name, bit=65,  prefix='RFFT_input1')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'RFFT',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def IRFFT(x,name=None):
    r""" Compute the inverse of the  DFT for real input.
    This function computes the inverse of the
    discrete Fourier Transform of real input computed by `rfft`.

    Args:
    ```text
    x : A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    out : A Variable `Tensor`. output shape is (n-1)*2
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((10,10,3),name='input0')
        out0 = sgs_chalk.IRFFT(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```

    """
    def isPower(n):
        if n <= 1:
            return False
        i = 1
        while i <= n:
            if i == n:
                return True
            i <<= 1
        return False
    #if len(x.shape) != 2:
    #    raise ValueError('FFT only support 2 dims')
    is_shape = (x.shape[-1]-1)*2
    if not isPower(is_shape):
        raise ValueError('IRFFT only support input number of 2^(n-1)+1 (n>0)')
    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    if is_shape >= 4:
        input1 = np.zeros((is_shape//4), dtype=np.complex64)
    else:
        input1 = np.zeros(int(is_shape), dtype=np.complex64)

    #cal output shape
    output_shape = [ ]
    input_shape = np.array(x.shape).tolist()
    for i in range(len(input_shape)):
        outputshape_1 = input_shape[i]
        output_shape.append(outputshape_1)
    output_shape[-1] = int((x.shape[-1]-1)*2)

    out0 = Tensor(shape=output_shape, dtype = 'float32', name=name, bit=33,  prefix='IRFFT')
    input1 = Tensor(data=input1, name=name, bit=65,  prefix='IRFFT_input1')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'IRFFT',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def FFT2(x, name=None):
    r"""Compute the 2-dimensional discrete Fourier Transform
    This function computes the *n*-dimensional discrete Fourier Transform
    over any axes in an *M*-dimensional array by means of the
    Fast Fourier Transform (FFT).  By default, the transform is computed over
    the last two axes of the input array, i.e., a 2-dimensional FFT. The last
    two axes must equal a power of 2, and not exceed 2 ** 13.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Complex64 Tensor.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    out: A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((128, 512), dtype='complex64', name='input')
        out0 = sgs_chalk.FFT2(in0, name='output')
        model = sgs_chalk.Model(in0, out0)
        model.save('test.sim')
    ```
    """

    if (len(x.shape)) < 2:
        raise ValueError('FFT2 need input shape at least two dimensions.')
    axes = [i for i, _ in enumerate(x.shape)]
    axes[-2], axes[-1] = axes[-1], axes[-2]
    out = Transpose(x, axes)
    out = FFT(out)
    out = Transpose(out, axes)
    out0 = FFT(out, name=name)
    return out0


@chalk_common.platform_register
def IFFT2(x, name=None):
    r"""Compute the 2-dimensional inverse discrete Fourier Transform.
    This function computes the inverse of the 2-dimensional discrete Fourier
    Transform over any number of axes in an M-dimensional array by means of
    the Fast Fourier Transform (FFT). By default, the inverse transform is
    computed over the last two axes of the input array. The last two axes
    must equal a power of 2, and not exceed 2 ** 13.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Complex64 Tensor.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    out: A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((128, 512), dtype='complex64', name='input')
        out0 = sgs_chalk.IFFT2(in0, name='output')
        model = sgs_chalk.Model(in0, out0)
        model.save('test.sim')
    ```
    """

    if (len(x.shape)) < 2:
        raise ValueError('IFFT2 need input shape at least two dimensions.')
    axes = [i for i, _ in enumerate(x.shape)]
    axes[-2], axes[-1] = axes[-1], axes[-2]
    out = Transpose(x, axes)
    out = IFFT(out)
    out = Transpose(out, axes)
    out0 = IFFT(out, name=name)
    return out0


@chalk_common.platform_register
def FLOAT2COMPLEX(x,name=None):
    r"""Use the input as the real part of a complex number and set the imaginary part to 0

    Args:
    ```text
    x : A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    out : A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.FLOAT2COMPLEX(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if x.minmax['bit'] != 33:
        raise ValueError(' only support input datatype float32')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='FLOAT2COMPLEX')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'FLOAT2COMPLEX',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def ExtractReal(x,name=None):
    r"""Take the real part of the input complex number as the output

    Args:
    ```text
    x : A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    out : A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.ExtractReal(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if x.minmax['bit'] != 65:
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, name=name, bit=33, prefix='ExtractReal')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'ExtractReal',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def AbsFp32(x, name=None):
    r"""
    Computes the absolute value of a tensor.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor` of same shape and type as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.AbsFp32(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    out0 = Tensor(shape=x.shape, name=name, bit=33, prefix='AbsFp32')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'AbsFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def SubFp32(x, y, name=None):
    r"""Returns x - y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.SubFp32(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.SubFp32(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.SubFp32(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=33)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32), bit=33)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))
    if in1.minmax['bit'] != 33 or in1.dtype != 'float32' or \
        x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')
    out0 = Tensor(shape=x.shape, name=name, bit=33, prefix='SubFp32')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'SubFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def MulFp32(x, y, name=None):
    r"""Returns x * y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.MulFp32(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.MulFp32(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.MulFp32(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=33)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32), bit=33)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))
    if in1.minmax['bit'] != 33 or in1.dtype != 'float32' or \
        x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')
    out0 = Tensor(shape=x.shape, name=name, bit=33, prefix='MulFp32')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'MulFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def DivFp32(x, y, name=None):
    r"""Returns x / y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.DivFp32(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.ones((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.DivFp32(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.DivFp32(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=33)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32), bit=33)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))
    if in1.minmax['bit'] != 33 or in1.dtype != 'float32' or \
        x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')
    out0 = Tensor(shape=x.shape, name=name, bit=33, prefix='DivFp32')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'DivFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def AddFp32(x, y, name=None):
    r"""Returns x + y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.AddFp32(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.AddFp32(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.AddFp32(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=33)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32), bit=33)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if in1.minmax['bit'] != 33 or in1.dtype != 'float32' or \
        x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')

    out0 = Tensor(shape=x.shape, name=name, bit=33,prefix='AddFp32')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'AddFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def SumFp32(x, axis, name=None):
    r"""cal sum vlaue of given axis.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,512),name='input0')
        out0 = sgs_chalk.SumFp32(in0,(1,2),name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if  x.minmax['bit'] != 33 or x.dtype != 'float32':
        raise ValueError(' only support input datatype float32')

    if not isinstance(axis, list) and not isinstance(axis, tuple) and not isinstance(axis, float) :
        raise ValueError('axis of SumFp32 only supports list or tuple')

    axis_1 = np.array(axis).tolist()
    check_add_one = lambda arr:functools.reduce(lambda x,y:(x+1==y if isinstance(x,int) else x[0] and x[1]+1==y, y),arr)[0]
    if check_add_one(axis_1) == False:
        raise ValueError('value of axis must be consistent')

    #cal output shape
    output_shape = np.array(x.shape).tolist()
    for i in axis_1:
        output_shape[i] = 1
    input1 = Tensor(data=np.array(axis_1).astype('int32'),  bit=32, prefix='axis')
    out0 = Tensor(shape=output_shape, name=name, bit=33,prefix='SumFp32')
    Operator([x,input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'SumFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def MulCN(x, y, name=None):
    r"""Returns x * y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor
       and the elements in the tensor are complex numbers.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        in1 = sgs_chalk.Input((28,512),dtype='complex64',name='input1')
        out0 = sgs_chalk.MulCN(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((28,512), dtype=np.complex64)
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.MulCN(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.MulCN(in0,5.0+1j,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=65)
    elif isinstance(y, complex):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.complex64), bit=65)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if in1.minmax['bit'] != 65 or in1.dtype != 'complex64' or \
        x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='MulCN')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'MulCN')
    return out0


@chalk_common.platform_register
def DivCN(x, y, name=None):
    r"""Returns x / y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor
       and the elements in the tensor are complex numbers.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        in1 = sgs_chalk.Input((28,512),dtype='complex64',name='input1')
        out0 = sgs_chalk.DivCN(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.ones((28,512), dtype=np.complex64)
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.DivCN(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.DivCN(in0,5.0+1j,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=65)
    elif isinstance(y, complex):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.complex64), bit=65)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if in1.minmax['bit'] != 65 or in1.dtype != 'complex64' or \
        x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='DivCN')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'DivCN')
    return out0

@chalk_common.platform_register
def ConjCN(x,name=None):
    r"""Conjugate transpose of complex numbers

    Args:
    ```text
    x : A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    out : A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.ConjCN(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='ConjCN')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'ConjCN',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def ConjMulCN(x, name=None):
    r"""Returns (real value of x)^2 + (image value of x)^2 + 0

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.ConjMulCN(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """

    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='ConjMulCN')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'ConjMulCN',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def AddCN(x, y, name=None):
    r"""Returns x + y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor
       and the elements in the tensor are complex numbers.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        in1 = sgs_chalk.Input((28,512),dtype='complex64',name='input1')
        out0 = sgs_chalk.AddCN(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        input1 = np.zeros((28,512), dtype=np.complex64)
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.AddCN(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.AddCN(in0,5.0+1j,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=65)
    elif isinstance(y, complex):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.complex64), bit=65)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if in1.minmax['bit'] != 65 or in1.dtype != 'complex64' or \
        x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='AddCN')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'AddCN',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def SubCN(x, y, name=None):
    r"""Returns x - y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and the elements in the tensor are complex numbers.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor
       and the elements in the tensor are complex numbers.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        in1 = sgs_chalk.Input((28,512),dtype='complex64',name='input1')
        out0 = sgs_chalk.SubCN(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')s

    case1:
        input1 = np.zeros((28,512), dtype=np.complex64)
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.SubCN(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        in0 = sgs_chalk.Input((28,512),dtype='complex64',name='input0')
        out0 = sgs_chalk.SubCN(in0,5.0+1j,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y, bit=65)
    elif isinstance(y, complex):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.complex64), bit=65)
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    if in1.minmax['bit'] != 65 or in1.dtype != 'complex64' or \
        x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')
    out0 = Tensor(shape=x.shape, dtype='complex64', name=name, bit=65, prefix='SubCN')
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'SubCN',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def TraceMatrix(x, name=None):
    r"""
    Computes the sum value of H&W.

    Given a tensor of integer or floating-point values, this operation returns a
    tensor of the same type, where each element contains the absolute value of the
    corresponding element in the input.

    Args:
    ```text
    x: A `Tensor`
    ```

    Returns:
    ```text
    A `Tensor`
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512,512,4),dtype='complex64',name='input0')
        out0 = sgs_chalk.TraceMatrix(in0,name='output')
        model = sgs_chalk.Model([in0],[out0])
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """

    if len(x.shape) < 3:
        raise ValueError(' len x.shape must >=3')

    if x.minmax['bit'] != 65 or x.dtype != 'complex64':
        raise ValueError(' only support input datatype complex64')

    #cal output shape
    output_shape = []
    for i in range(len(x.shape)):
        if i == len(x.shape)-3 or i == len(x.shape)-2:
            value = 1
            output_shape.append(value)
        else:
            value = x.shape[i]
            output_shape.append(x.shape[i])
    out0 = Tensor(shape=output_shape, dtype='complex64', name=name, bit=65, prefix='TraceMatrix')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'TraceMatrix',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0


@chalk_common.platform_register
def Matmul(x, y, name=None):
    r"""The inputs must, following any transpositions, be tensors of rank >= 2
    where the inner 2 dimensions specify valid matrix multiplication arguments,
    and any further outer dimensions match.
    Both matrices must be of the same type. The supported type is  `float32`

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor and shape is (h,w,n).
    y: A `Tensor`. Must be Variable Tensor and shape is (w,c,n).
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    `Tensor` of the same type as `a` and `b` where each inner-most matrix is
    the product of the corresponding matrices in `a` and `b`
    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
    for all indices i, j.
    In this op,shape is (h,c,n)

    Note: This is matrix product, not element-wise product.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((52,28,512),dtype='float32',name='input0')
        in1 = sgs_chalk.Input((28,114,512),dtype='float32',name='input1')
        out0 = sgs_chalk.Matmul(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')
    ```
    """
    if len(x.shape) != 3 or len(y.shape) != 3:
        raise ValueError('len input shape must be 3,Matmul error,')
    if x.shape[1] != y.shape[0]:
        raise ValueError('x.shape[1] != y.shape[0],Matmul error,')
    if x.minmax['bit'] != 33 or x.dtype != 'float32' or \
        y.minmax['bit'] != 33 or y.dtype != 'float32':
        raise ValueError(' only support input datatype float32')
    out0 = Tensor(shape=(x.shape[0],y.shape[1],x.shape[2]), name=name, bit=33, prefix='MatmulFp32')
    Operator([x,y], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'MatmulFp32',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def MedianFilter(x, filtertype,filtersize, name=None):
    r"""
    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor
    filtertype: only support MID/MAX/MIN
    filtersize: only support 3X3/5X5
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,4),dtype='int16',name='input0')
        out0 = sgs_chalk.MedianFilter(in0,'MAX','5X5',name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)

    case1:
        in0 = sgs_chalk.Input((1,9,9,1),dtype='int16',name='input0')
        out0 = sgs_chalk.MedianFilter(in0,'MAX','3X3',name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """


    if filtertype == 'MID':
        filter_type = 0
    elif filtertype == 'MAX':
        filter_type = 1
    elif filtertype == 'MIN':
        filter_type = 2
    else:
        raise ValueError('not support this filtertype')

    if filtersize == '3X3':
        filter_size = 0
        filter_w = 3
        filter_h = 3
    elif filtersize == '5X5':
        filter_size = 1
        filter_w = 5
        filter_h = 5
    else:
        raise ValueError('not support this filtersize')

    MedianFilter_custom_options = [
     (b"filterw",filter_w,"int"),
     (b"filterh",filter_h,"int"),
     (b"enfiltertype",filter_type,"int"),
     (b"enfiltersize",filter_size,"int")
    ]

    out0 = Tensor(shape=(x.shape), name=name, prefix='MedianFilter')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'MedianFilter',
            builtin_options_type=None,builtin_options=None,custom_options=MedianFilter_custom_options)
    return out0

@chalk_common.platform_register
def ConvFilter(x, weight, bias = None, Activation = 'NONE', stride = (1,1), padding = (0,0,0,0), dilation = (1,1), name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor  C of input must be 1
    weight: filters of shape(out_channels, kH, kW, in_channels)  out_channels must be 1
    bias: optional bias tensor of shape(out_channels).Default:None
    stride: the stride of the convolving kernel. Can be a single number or a tuple(sH,sW).Default: 1
    padding: a tuple (paddingLeft, paddingRight, paddingTop, paddingBottom).
    dilation: the spacing between kernel elements. Can be a single number or a tuple(dH,dW).Default: 1
    Activation: only support NONE/RELU/RELU_N1_TO_1/RELU6
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((1,28,28,1),dtype='int16',name='input0')
        input = [1,3,2,4,5,6,7,8,9]
        input1 = np.reshape(input,(1,3,3,1))
        out0 = sgs_chalk.ConvFilter(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        #model.save('test.sim')
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if weight.shape[0] != 1:
        raise ValueError('N of weight must be 1')
    if weight.shape[3] != 1 or x.shape[3] != 1:
        raise ValueError('C of weight && input must be 1')
    if weight.shape[2] > 9:
        raise ValueError('W of weight must less than 9')
    if (x.shape[2] + padding[0] + padding[1]) < 9:
        raise ValueError('x.w + Pad.Left + Pad.Right must greater than 9')


    if isinstance(weight, np.ndarray):
        if weight.astype != 'float32':
            weight = weight.astype('float32')
        weight = Tensor(data=weight)
    elif isinstance(weight, Tensor):
        weight = weight
    else:
        raise ValueError('Not support this type:', type(weight))

    if weight.is_const != True:
        raise ValueError('weight tensor only support const tensor')

    if bias == None:
        bias = np.zeros((weight.shape[0]), dtype=np.float32)
        bias = Tensor(data=bias, name='bias', bit=bit, prefix='bias')
    if isinstance(bias, np.ndarray):
        bias = Tensor(data=bias,name='bias')
    elif isinstance(bias, Tensor):
        bias = bias
    else:
        raise ValueError('Not support this type:', type(bias))

    padding_1 = padding
    kernel_size = [weight.shape[1],weight.shape[2]]
    strideW = stride[0]
    strideH = stride[1]
    dilationWFactor = dilation[0]
    dilationHFactor = dilation[1]
    paddingLeft = padding_1[0]
    paddingRight = padding_1[1]
    paddingTop = padding_1[2]
    paddingBottom = padding_1[3]

    #SAME = 0  VALID = 1  CAFFE = 2
    padding =2

    if Activation == 'NONE':
        fusedActivationFunction = 0
    elif Activation == 'RELU':
        fusedActivationFunction = 1
    elif Activation == 'RELU_N1_TO_1':
        fusedActivationFunction = 2
    elif Activation == 'RELU6':
        fusedActivationFunction = 3
    else:
        raise ValueError('Not support this activation !!!')

    output_shape_H = int((x.shape[1] + 2 * padding_1[0] - dilation[0] * (kernel_size[0]-1) -1) / stride[0]) + 1
    output_shape_W = int((x.shape[2] + 2 * padding_1[1] - dilation[1] * (kernel_size[1]-1) -1) / stride[1]) + 1
    output_shape_C = weight.shape[0]
    output_shape = [1, output_shape_H, output_shape_W, output_shape_C]

    ConvFilter_custom_options = [
     (b"Padding",padding,"int"),
     (b"StrideW",strideW,"int"),
     (b"StrideH",strideH,"int"),
     (b"FusedActivationFunction",fusedActivationFunction,"int"),
     (b"DilationWFactor",dilationWFactor,"int"),
     (b"DilationHFactor",dilationHFactor,"int"),
     (b'PaddingLeft',paddingLeft,"int"),
     (b'paddingRight',paddingRight,"int"),
     (b'PaddingTop',paddingTop,"int"),
     (b'PaddingBottom',paddingBottom,"int")
    ]

    out0 = Tensor(shape = output_shape, name=name, prefix='ConvFilter')
    Operator([x,weight,bias], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM,custom_code=b'ConvFilter',
            builtin_options_type=None,builtin_options=None,custom_options=ConvFilter_custom_options)
    return out0

@chalk_common.platform_register
def Elu(x,alpha=1.0,name=None,bit=16,minimum=None,maximum=None):
    r"""
    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    name: A name for the model Input tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as settings.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Elu(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    Elu_custom_options = [
     (b"alpha",alpha,"float")
    ]
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Elu')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().ELU,
            custom_options=Elu_custom_options)
    return out0

@chalk_common.platform_register
def CustomizedScatterND(input, indices, update, reduction, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    input: A `Tensor`. Must be Variable Tensor whose rank r >= 1.
    indices: A const Tensor or numpy.array whose rank q >= 1.
    update: A Tensor whose rank is q+r-indices.shape[-1]-1
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        input = sgs_chalk.Input((14,3,41,2,1,2,3,5), name='input')
        indices = np.array([[5, 0, 1], [10, 2, 33]])
        update = sgs_chalk.Input((2,2,1,2,3,5), name='update')
        scatterND_out = sgs_chalk.CustomizedScatterND(input, indices, update, 0, name='scatterND_out')
        model = sgs_chalk.Model([input,update], scatterND_out)
        model.save('test.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """
    if isinstance(indices, np.ndarray):
        if indices.dtype != 'int32':
            indices = indices.astype('int32')
        indices = Tensor(data=indices, name='indices', bit=32)
    elif isinstance(indices, Tensor):
        indices = indices
    else:
        raise ValueError('Not support this type:', type(indices))
    if reduction != 0:
        raise ValueError('reduction must be 0(none)')
    ScatterNd_options = [
     (b"reduction", reduction, "int"),
    ]
    out0 = Tensor(shape=input.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='CustomizedScatterND')
    Operator([input, indices, update], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'Customized_ScatterND',
            builtin_options_type=None,builtin_options=None,custom_options=ScatterNd_options)
    return out0

@chalk_common.platform_register
def Softplus(input, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    input: A `Tensor`. Must be Variable Tensor.
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        input = sgs_chalk.Input((14,3,41,2,1,2,3,5), name='input')
        softplus_out = sgs_chalk.Softplus(input, name='softplus_out')
        model = sgs_chalk.Model([input,], softplus_out)
        model.save('test.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """

    out0 = Tensor(shape=input.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Softplus')
    Operator([input], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'Softplus',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def Square(input, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    input: A `Tensor`. Must be Variable Tensor.
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        input = sgs_chalk.Input((14,3,41,2,1,2,3,5), name='input')
        square_out = sgs_chalk.Square(input, name='softplus_out')
        model = sgs_chalk.Model([input,], square_out)
        model.save('test.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """
    out0 = Tensor(shape=input.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Square')
    Operator([input], [out0], tflite.BuiltinOperator.BuiltinOperator().SQUARE,
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def Atan2(input0, input1, name=None):
    r"""Element-wise arctangent of `input0` and `input1` with consideration of the quadrant.
    Returns a new tensor with the signed angles in radians between (-pi, pi)

    Args:
    ```text
    input:0 A `Tensor`. Must be Variable Tensor.
    input1: A `Tensor`. Must be Variable Tensor.
    name: A name for the model Output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        input0 = sgs_chalk.Input((1, 22535), name='input0')
        input1 = sgs_chalk.Input((1, 22535), name='input1')
        out = sgs_chalk.Atan2(input0, input1, name='output')
        model = sgs_chalk.Model([input0, input1], out)
        model.save('atan2.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """

    if not isinstance(input0, Tensor):
        raise ValueError('Not support this type:', type(input0))
    if not isinstance(input1, Tensor):
        raise ValueError('Not support this type:', type(input1))
    if input0.shape != input1.shape:
        raise ValueError('input0 ande input1 must be same shape!')
    if (input0.minmax['min'], input0.minmax['max']) != ([-32767], [32767]) or \
        (input1.minmax['min'], input1.minmax['max']) != ([-32767], [32767]):
        raise ValueError('input0 and input1 [min, max] must be [-32767, 32767]')

    out0 = Tensor(shape=input0.shape, name=name, bit=16, minimum=-3.141592653589793, maximum=3.141592653589793, prefix='Atan2')
    Operator([input0, input1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'Atan2',
            builtin_options_type=None, builtin_options=None, custom_options=None)
    return out0

@chalk_common.platform_register
def ReduceMin(input, axis, keepdims=1, name=None, bit=16, minimum=None, maximum=None):
    r"""

    Args:
    ```text
    input: A `Tensor` or 'Array'. Must be Variable Tensor if it's a Tensor.
    asix: A `Tensor` or 'Array'. Must be Const Tensor if it's a Tensor.
    keepdims: A one dimension array or list.
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        in0 = sgs_chalk.Input((1,28,512,33),name='input0')
        axis = np.array([-2,1],dtype=np.int32)
        out0 = sgs_chalk.ReduceMin(in0, axis, keepdims=1, name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('reducemin.sim',input_config='./input_config.ini', convert_fixed=True, inputs='RAWDATA')
    ```
    """
    if isinstance(input, np.ndarray):
        input = Tensor(data=input, name='input')
    elif isinstance(input, Tensor):
        input = input
    else:
        raise ValueError('Not support this type:', type(input))
    if isinstance(axis, list):
        axis = np.array(axis, dtype='int32')
    elif isinstance(axis, np.ndarray):
        if axis.dtype != 'int32':
            axis = axis.astype('int32')
    else:
        raise ValueError('Not support this type:', type(axis))

    if len(axis.shape) != 1:
        raise ValueError('axis must be one dimension array/list!')
    dim_size = len(input.shape)
    axis_temp = np.zeros(axis.shape, dtype=axis.dtype)
    for i in range(len(axis)):
        axis_temp[i] = axis[i] if axis[i] >= 0 else axis[i] + dim_size
    axis_temp = np.unique(sorted(axis_temp))
    output_shape = []
    for i in range(dim_size):
        is_axis = False
        for j in range(len(axis_temp)):
            if i == axis_temp[j]:
                is_axis = True
                break
        if is_axis:
            if keepdims != 0:
                output_shape.append(1)
        else:
            output_shape.append(input.shape[i])
    output_shape = np.array(output_shape)
    print("output shape is", output_shape)
    tflite.ReducerOptions.ReducerOptionsStart(BUILDER)
    tflite.ReducerOptions.ReducerOptionsAddKeepDims(BUILDER, keepdims)
    ReduceMin_options = tflite.ReducerOptions.ReducerOptionsEnd(BUILDER)
    out0 = Tensor(shape=output_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='ReduceMin')
    Operator([input, Tensor(data=axis, name='axis', bit=32)], [out0], tflite.BuiltinOperator.BuiltinOperator().REDUCE_MIN,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().ReducerOptions,builtin_options=ReduceMin_options)
    return out0
@chalk_common.platform_register
def CustomPow(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""

    Args:
    ```text
    input: x `Tensor`. Must be Variable Tensor.
    input: y `Tensor`. Must be Constant Scaler.
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        input = sgs_chalk.Input((14,3,41,2,1,2,3,5), name='input')
        input1 = sgs_chalk.Input(0.3333333333333333, name='input1')
        customPow_out = sgs_chalk.CustomPow(input,input1, name='customPow_out')
        model = sgs_chalk.Model([input,], customPow_out)
        model.save('test.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """
    if isinstance(y, np.float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    else:
        raise ValueError('Not support this type:', type(y))
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='CustomPow')
    #create option
    cus_options = [(b"numeratort",1,"int")]
    Operator([x,in1], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'CustomPow',
            builtin_options_type=None,builtin_options=None,custom_options=cus_options)
    return out0

@chalk_common.platform_register
def Div(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns x / y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.Div(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Div(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.ones((28,512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Div(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.DivOptions.DivOptionsStart(BUILDER)
    tflite.DivOptions.DivOptionsAddFusedActivationFunction(BUILDER,0)
    div_options = tflite.DivOptions.DivOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Div')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().DIV,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().DivOptions,builtin_options=div_options)
    return out0

@chalk_common.platform_register
def Sin(x, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns sin(x) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Sin(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')


    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Sin')
    #cus_options = [(b"numeratort",1,"int")]
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().SIN)
    return out0

@chalk_common.platform_register
def Cos(x, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns Cos(x) element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.Sin(in0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')


    """
    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='Cos')
    Operator([x], [out0], tflite.BuiltinOperator.BuiltinOperator().COS)
    return out0

def DmaCoefMatrix(input, name=None, bit=16, minimum=None, maximum=None):
    r"""
    Args:
    ```text
    input: input `Tensor`. Must be Variable Tensor.
    name: A name for the model output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as input.
    ```

    Examples:
    ```python
    case0:
        from calibrator_custom import sgs_chalk
        import numpy as np
        input = sgs_chalk.Input((1,75,75,3), name='input')
        dmacoefmatrix_out = sgs_chalk.DmaCoefMatrix(input, name='dmacoefmatrix_out')
        model = sgs_chalk.Model([input], dmacoefmatrix_out)
        model.save('test.sim', input_config='./input_config.ini', convert_fixed=True)
    ```
    """
    if not isinstance(input, Tensor):
        raise ValueError('Not support this type:', type(input))
    out_shape=[]
    length=len(input.shape)
    for i in range(length):
        out_shape.append(input.shape[i])
    out_shape[length-1]=4
    print(out_shape)
    out0 = Tensor(shape=out_shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='DmaCoefMatrix')
    Operator([input], [out0], tflite.BuiltinOperator.BuiltinOperator().CUSTOM, custom_code=b'DmaCoefMatrix',
            builtin_options_type=None,builtin_options=None,custom_options=None)
    return out0

@chalk_common.platform_register
def FloorDiv(x, y, name=None, bit=16, minimum=None, maximum=None):
    r"""Returns x / y element-wise.

    Args:
    ```text
    x: A `Tensor`. Must be Variable Tensor.
    y: A `Tensor` or `numpy.ndarray`. Must have the same type as `x`, can be Variable or Const Tensor.
       Support inner most dimension broadcasting.
    name: A name for the output tensor (optional).
    ```

    Returns:
    ```text
    A Variable `Tensor`. Has the same shape as `x`.
    ```

    Examples:
    ```python
    case0:
        in0 = sgs_chalk.Input((28,512),name='input0')
        in1 = sgs_chalk.Input((28,512),name='input1')
        out0 = sgs_chalk.FloorDiv(in0,in1,name='output')
        model = sgs_chalk.Model([in0,in1],out0)
        model.save('test.sim')

    case1:
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.FloorDiv(in0,5.0,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim')

    case2:
        input1 = np.ones((512), dtype=np.float32)
        in0 = sgs_chalk.Input((28,512),name='input0')
        out0 = sgs_chalk.FloorDiv(in0,input1,name='output')
        model = sgs_chalk.Model([in0],out0)
        model.save('test.sim',input_config='./input_config.ini',convert_fixed=True)
    ```
    """
    if isinstance(y, np.ndarray):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = Tensor(data=y)
    elif isinstance(y, float):
        chalk_common.check_elementwise_tensor_shape(x.shape, np.array([y]))
        in1 = Tensor(data=np.array([y]).astype(np.float32))
    elif isinstance(y, Tensor):
        chalk_common.check_elementwise_tensor_shape(x.shape, y)
        in1 = y
    else:
        raise ValueError('Not support this type:', type(y))

    tflite.FloorDivOptions.FloorDivOptionsStart(BUILDER)
    div_options = tflite.FloorDivOptions.FloorDivOptionsEnd(BUILDER)

    out0 = Tensor(shape=x.shape, name=name, bit=bit, minimum=minimum, maximum=maximum, prefix='FloorDiv')
    Operator([x, in1], [out0], tflite.BuiltinOperator.BuiltinOperator().FLOOR_DIV,
            builtin_options_type=tflite.BuiltinOptions.BuiltinOptions().FloorDivOptions,builtin_options=div_options)
    return out0

