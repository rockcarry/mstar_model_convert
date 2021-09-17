#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import tensorflow.contrib.slim as slim
import copy
import argparse


support_operations=[u'NoOp', u'Placeholder', u'Const', 
u'Pad', u'Identity', u'Add', u'Rsqrt', u'Mul', u'Sub', 
u'FakeQuantWithMinMaxVars', u'Conv2D', u'BiasAdd', 
u'MaxPool', u'FusedBatchNorm', u'Relu', u'Mean', 
u'Squeeze', u'Reshape', u'Softmax', u'Shape',
u'StridedSlice',u'Pack',u'Max',u'Exp',u'RealDiv', u'ConcatV2',
u'Minimum', u'TopKV2', u'GatherV2', u'Split', u'Maximum',
u'NonMaxSuppressionV3', u'Sqrt',u'Round', u'Cast', u'Equal',
u'Range', u'ExpandDims', u'MatMul',u'Less', u'LogicalAnd',
u'Sigmoid',u'Assert', u'GreaterEqual', u'Relu6', u'DepthwiseConv2dNative', 
u'SpaceToBatchND', u'BatchToSpaceND', u'AvgPool',u'Slice',
u'ZerosLike',u'Transpose',u'Size',u'Rank',u'Tile',
u'Greater',u'Unpack',u'Gather',u'Fill']


def arg_parse():
    parser = argparse.ArgumentParser(description='ops_check')
    parser.add_argument('pb_path', type=str,
                        help='tensorflow pb path')
    return parser.parse_args()


def get_ops_type_from_tensorflow_pb(pb_path="the frozen pb path") :
    """
    args : the tensorflow pb address
    return :operation attributes list
    """

    with tf.compat.v1.Session() as sess:
        # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')
        tf.compat.v1.global_variables_initializer()
        output_graph_def = tf.compat.v1.GraphDef()  
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        #totalsize = slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)
        print ("\n\n\n\n\n")
        print ("op counts =",len(tf.compat.v1.get_default_graph().get_operations()))
        ops_type = []
        for op in tf.compat.v1.get_default_graph().get_operations():
            #print (op.outputs)
            #print ("\n\n\n")
            """
            print ("type:",op.type)
            print ("name",op.name)
            print ("outputs",op.outputs)
            for i in range(len(op.inputs)):
                    print  ("inputs :",i,":",op.inputs[i])
                    #print  (op.inputs[i].shape)
            print ("\n\n")
            """

            #print ("\n\n\n")
            if (op.type not in ops_type) :
                ops_type.append(op.type)   

        print ("pb's ops:")
        print (ops_type)
        unsupport_operations=[]
        for operation in ops_type :
            if (operation not in support_operations):
                unsupport_operations.append(operation)
        print ("unsupport_operations ops:")
        print (unsupport_operations)
    return ops_type







if __name__ == "__main__" :
    pb_path=arg_parse().pb_path
    get_ops_type_from_tensorflow_pb(pb_path)
