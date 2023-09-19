import calibrator_custom
import random as random
import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import six
import numpy as np
import pdb

def arg_parse():
    parser = argparse.ArgumentParser(description='Show tensor info Tool')
    subparsers = parser.add_subparsers(help='Parameter usage instructions')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help=' Fixed Model path.')
    parser.add_argument('-u', '--unit', type=str, required=False, default=None,
                        help=' Select units: If this parameter is configured as "op", it will be expressed in units of op (default in units of tensor).')
    return parser.parse_args()

def get_fig_size_op(modelInfo):
    tensor_allocs = modelInfo.get_op_allocs()
    # all tensor info
    op_tensor_list = []
    op_top_max = 0
    for i,info in enumerate(tensor_allocs):
        for j in six.moves.range(len(info['op_input_alloclist'])):
            tensor_info = info['op_input_alloclist'][j]
            op_tensor_list.append(tensor_info['name'])
            if (tensor_info['alloc_ended']/(1024*1024)) > op_top_max:
                op_top_max = round(tensor_info['alloc_ended']/(1024*1024))
        for j in six.moves.range(len(info['op_output_alloclist'])):
            tensor_info = info['op_output_alloclist'][j]
            op_tensor_list.append(tensor_info['name'])
            if (tensor_info['alloc_ended']/(1024*1024)) > op_top_max:
                op_top_max = round(tensor_info['alloc_ended']/(1024*1024))
    # fig set
    x_len = len(op_tensor_list)
    size_w = (x_len/100)*14
    size_h = op_top_max + 2
    return size_w,size_h,op_tensor_list

def get_fig_size_tensor(modelInfo):
    tensor_allocs = modelInfo.get_tensor_allocs()
    # all tensor info
    op_tensor_list = []
    op_top_max = 0
    for i,info in enumerate(tensor_allocs):
        if info['name'] not in op_tensor_list:
            op_tensor_list.append(info['name'])
        if (info['alloc_ended']/(1024*1024)) > op_top_max:
            op_top_max = round(info['alloc_ended']/(1024*1024))

    # fig set
    x_len = len(op_tensor_list)
    size_w = (x_len/100)*15
    size_h = op_top_max + 2
    return size_w,size_h,op_tensor_list

def get_fig_color_op(tensorName):
    tensor_color = {}
    color_record = []
    for i in six.moves.range(len(tensorName)):
        r = random.random()
        b = random.random()
        g = random.random()
        random_color = (r,g,b)
        if random_color not in color_record:
            color_record.append(random_color)
            tensor_color[tensorName[i]] = random_color
    return tensor_color

def get_fig_color_tensor(tensorName):
    tensor_color = cm.Set3(np.arange(3) / 5)
    return tensor_color

def get_mapping_table_op(tensorName):
    tensor_id = {}
    count = 0
    for i, item in enumerate(tensorName):
        if item not in tensor_id.keys():
            tensor_id[item] = count
            count += 1

    save_txt = os.path.join(os.getcwd(), 'TensorID_op'+ '.txt')
    with open(save_txt, 'w') as f:
        for i in six.moves.range(len(tensorName)):
            data = f.write('TensorID——>' + '[' + str(tensor_id[tensorName[i]]) + ']' + ':TensorName——>' + str(tensorName[i])+ "\n" )
    print("=== create TXT done, file [TensorID.txt] at:  %s " % (save_txt))
    return tensor_id

def get_mapping_table_tensor(tensorName):
    save_txt = os.path.join(os.getcwd(), 'TensorID_tensor'+ '.txt')
    with open(save_txt, 'w') as f:
        for i in six.moves.range(len(tensorName)):
            data = f.write( 'TensorID——>' + '[' + str(i) + ']' + ':TensorName——>' + str(tensorName[i])+ "\n" )
    print("=== create TXT done, file [TensorID.txt] at:  %s " % (save_txt))
    return

def draw_TensorAllocationInformation_op(modelInfo,tensor_color,tensor_id,size_h,all_tensor):
    tensor_allocs = modelInfo.get_op_allocs()
    max_value = 0
    sub_start,sub_end = 0,0
    for i,info in enumerate(tensor_allocs):
        op_name = info['op_name']
        op_index = info['op_index']
        op_tensor_list,op_tensor_height_list,op_tensor_bottom_list,op_tensor_color_list,op_top_list = [],[],[],[],[]
        # op input info
        op_input_name,op_input_top,op_input_height,op_input_bottom,op_input_color = [],[],[],[],[]
        op_output_name,op_output_top,op_output_height,op_output_bottom,op_output_color = [],[],[],[],[]
        # get op input
        for j,tensor_info in enumerate(info['op_input_alloclist']):
            op_input_name.append(tensor_info['name'])
            op_input_top.append(tensor_info['alloc_ended']/(1024*1024))
            op_input_height.append(tensor_info['alloc_ended']/(1024*1024) - tensor_info['alloc_started']/(1024*1024))
            op_input_bottom.append(tensor_info['alloc_started']/(1024*1024))
            # get tensor color
            op_input_color.append(tensor_color[tensor_info['name']])
            if tensor_info['alloc_ended']/(1024*1024) > max_value:
                max_value = tensor_info['alloc_ended']/(1024*1024)
        # get op output
        for j,tensor_info in enumerate(info['op_output_alloclist']):
            op_output_name.append(tensor_info['name'])
            op_output_top.append(tensor_info['alloc_ended']/(1024*1024))
            op_output_height.append(tensor_info['alloc_ended']/(1024*1024) - tensor_info['alloc_started']/(1024*1024))
            op_output_bottom.append(tensor_info['alloc_started']/(1024*1024))
            # get tensor color
            op_input_color.append(tensor_color[tensor_info['name']])
            if tensor_info['alloc_ended']/(1024*1024) > max_value:
                max_value = tensor_info['alloc_ended']/(1024*1024)

        op_tensor_list = op_input_name + op_output_name
        op_top_list = op_input_top + op_output_top
        op_tensor_height_list = op_input_height + op_output_height
        op_tensor_bottom_list = op_input_bottom + op_output_bottom
        op_tensor_color_list = op_input_color + op_output_color
        # create subplot
        # sub_num = i + 1
        # ax= plt.subplot(1,len(tensor_allocs),sub_num)
        # ax.margins(0.02)
        sub_start = sub_end
        sub_end = sub_start + len(op_tensor_list)
        gs = GridSpec(1, len(all_tensor))
        ax = fig.add_subplot(gs[0, sub_start:sub_end])
        ax.margins(0.02)
        x = [i for i in six.moves.range(len(op_tensor_list))]
        ax.bar(x, op_tensor_height_list, width=0.6, bottom = op_tensor_bottom_list,align='center',color = op_tensor_color_list)
        ax.set_ylim(0,size_h)
        ax.set_title(op_index,loc='center',fontsize= 10,y=-0.04)
        plt.xticks(rotation = 90,fontsize= 6)
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
        ax.axes.xaxis.set_visible(False)
        for a,b,i in zip(x,op_top_list,range(len(x))):
            if op_top_list[i] != 0:
                plt.text(a,b+0.1,"%.2f"%op_top_list[i],ha='center',verticalalignment='bottom',fontsize=7,rotation = 90,color = 'darkslategray')
        for a,b,name in zip(x,[0.02 for i in six.moves.range(len(x))],op_tensor_list):
            plt.text(a,b+0.01,"%d"%tensor_id[name],ha='center',verticalalignment='bottom',fontsize=6,rotation = 90,color = 'darkorange')

    fig.tight_layout(pad=0.4, w_pad=0, h_pad=0)
    fig.suptitle('Tensor Allocation Information (MB)',fontweight ="bold")
    plt.subplots_adjust(top=0.90)
    plt.savefig(save_name)
    print("=== MAX Tensor Allocation Value is:  %.2f MB" % (max_value))
    return

def draw_TensorAllocationInformation_tensor(modelInfo,tensor_color):
    tensor_allocs = modelInfo.get_tensor_allocs()
    name_list = []
    bottom_list = []
    top_list = []
    height_list = []
    max_value = 0
    for i in six.moves.range(len(tensor_allocs)):
        alloc = tensor_allocs[i]
        name_list.append(alloc['name'])
        bottom_list.append(alloc['alloc_started']/(1024*1024))
        top_list.append(alloc['alloc_ended']/(1024*1024))
        height_list.append(alloc['alloc_ended']/(1024*1024) - alloc['alloc_started']/(1024*1024))
        if alloc['alloc_ended']/(1024*1024) > max_value:
            max_value = alloc['alloc_ended']/(1024*1024)

    plt.xlabel("TensorID")
    x_len = len(tensor_allocs)
    my_x_ticks = np.arange(0, x_len)
    plt.xticks(my_x_ticks,rotation = 90,fontsize= 7)
    x = [i for i in six.moves.range(len((tensor_allocs)))]
    plt.bar(x, height_list, width=0.8, bottom = bottom_list,align='center',color = tensor_color)
    for a,b,i in zip(x,top_list,range(len(x))):
        plt.text(a,b+0.01,"%.2f"%top_list[i],ha='center',verticalalignment='bottom',fontsize=7,rotation = 90)

    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.grid(ls='--')
    plt.legend(['Tensor memory  (MB)'],loc = 'upper center')
    #plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.1,hspace=0.1)
    plt.title('Tensor Allocation Information ')
    print("=== MAX Tensor Allocation Value is:  %.2f MB" % (max_value))
    plt.savefig(save_name)
    return

if __name__ == '__main__':
    args = arg_parse()
    model_path = args.model
    unit_choice = args.unit
    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))
    print("-------------------------START-------------------------")
    model = calibrator_custom.fixed_simulator(model_path)
    if model.class_type != 'FIXED':
        raise ValueError('Expected {} is Fixed model'.format(model_path))
    if calibrator_custom.utils.get_sdk_version() not in ['1', 'Q_']:
        raise ValueError('Unsupported SDK version {}'.format(calibrator_custom.__version__))

    if unit_choice == 'op':
        save_name = os.path.join(os.getcwd(), 'TensorAllocationInformation_op_Fixed'+ '.png')
        # set fig size
        fig_W,fig_H,all_tensor = get_fig_size_op(model)
        fig = plt.figure(dpi=100,figsize=(fig_W,fig_H))
        # set tensor color
        my_color = get_fig_color_op(all_tensor)
        # generate mapping table
        my_id = get_mapping_table_op(all_tensor)
        # get op info
        draw_TensorAllocationInformation_op(model,my_color,my_id,fig_H,all_tensor)
    elif unit_choice == 'tensor' or unit_choice is None:
        save_name = os.path.join(os.getcwd(), 'TensorAllocationInformation_tensor_Fixed'+ '.png')
        # set fig size
        fig_W,fig_H,all_tensor = get_fig_size_tensor(model)
        fig = plt.figure(dpi=100,figsize=(fig_W,fig_H))
        # set tensor color
        my_color = get_fig_color_tensor(all_tensor)
        # generate mapping table
        get_mapping_table_tensor(all_tensor)
        # get op info
        draw_TensorAllocationInformation_tensor(model,my_color)
    print("=== create PNG done, file [TensorAllocationInformation.png] at:  %s " % (save_name))
    print("-------------------------END-------------------------")