# -*- coding: utf-8 -*-

import argparse
import os

import calibrator_custom
from calibrator_custom import utils
from calibrator_custom.ipu_quantization_lib import get_quant_config
from calibrator_custom.ipu_quantization_lib import quantize_pipeline, run_mpq, auto_pipeline
from calibrator_custom.ipu_quantization_lib import run_qat

import torch
import torch.backends.cudnn as cudnn
import pickle
import pdb
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Net(calibrator_custom.SIM_Calibrator):
    def __init__(self, model_path, input_config, core_mode, log=False):
        super().__init__()
        self.model = calibrator_custom.calibrator(model_path, input_config,
                                                  work_mode=core_mode, show_log=log
                                                  )
        self.chip_version = calibrator_custom.__version__

    def forward(self, x):
        in_details = self.model.get_input_details()
        out_details = self.model.get_output_details()
        for idx, _ in enumerate(in_details):
            self.model.set_input(idx, x[idx])
        self.model.invoke()
        result_list = []
        for idx, _ in enumerate(out_details):
            result = self.model.get_output(idx)
            result_list.append(result)
        return result_list

    def get_fixed_model(self, config, output_name=None, num_process=10, quant_level="L5"):
        out_model = utils.get_out_model_name(config.onnx_file, output_name)
        # print('\033[92mFeed sim_params\033[0m')
        if config.sim_params_file is not None:
            sim_params = pickle.load(open(config.sim_params_file, 'rb'))
        else:
            sim_params=None

        if config.onnx_file.endswith('_float.sim') and sim_params is not None:
            self.convert(quant_param=sim_params, fix_model=[out_model])
        else:
            preprocess_funcs = [utils.image_preprocess_func(n) for n in config.python_file.split(',')]
            # image_path = config.calset_dir
            if config.image_list is None:
                image_list = utils.all_path(config.calset_dir)
                img_gen = utils.image_generator(image_list, preprocess_funcs)
            elif os.path.basename(config.image_list).split('.')[-1].lower() in utils.image_suffix:
                img_gen = utils.image_generator([config.calset_dir], preprocess_funcs)
            else:
                with open(config.image_list, 'r') as f:
                    multi_images = f.readlines()
                multi_images = [[os.path.join(config.calset_dir, i) for i in images.strip().split(',')] for images in multi_images]
                img_gen = utils.image_generator(multi_images, preprocess_funcs)

            self.convert(img_gen, num_process=num_process, quant_level=quant_level,
                        quant_param=sim_params, fix_model=[out_model])
        print('\033[92mFixed model generated!\033[0m')

        return None

def set_quant_config(q_mode,quant_cfg,net,args):
    if q_mode is None or q_mode == 'Q0':
        print('Import quantization parameters.')
        quant_cfg.sim_params_file = args.q_param
    elif q_mode == 'Q10':
        print('Full 16bit quantization')
        quant_cfg.ib_quant_method = 'simple_minmax'
        quant_loss = quantize_pipeline(
            quant_cfg, net, quant_precision=16
        )
    elif q_mode == 'Q11' or q_mode == 'Q1':
        print('Full 8bit quantization')
        quant_cfg.ib_quant_method = 'simple_minmax'
        quant_loss = quantize_pipeline(
            quant_cfg, net, quant_precision=8
        )
    elif q_mode == 'Q12':
        print('Full 8bit omse-quantization')
        quant_cfg.ib_quant_method = 'omse'
        quant_loss = quantize_pipeline(
            quant_cfg, net, quant_precision=8,
            retrain_mpq_mode=2, fast_mpq_8_16=True
        )
    elif q_mode == 'Q13':
        if not hasattr(quant_cfg, 'mixed_precisions'):
            quant_cfg.mixed_precisions = [4, 8]
        print('MPQ', quant_cfg.mixed_precisions, 'quantization')
        quant_cfg.ib_quant_method = 'omse'
        sim_params = run_mpq(
            quant_cfg, net,# size_flag=0,
            mp_rate=quant_cfg.mp_rate,
            retrain=0
        )
        quant_cfg.sim_params_file = sim_params
        quant_cfg.torch_params_file = sim_params.replace('simx_', 'torchx_')
    elif q_mode == 'Q20':
        print('AdaQuantE 16bit quantization')
        quant_cfg.ib_quant_method = 'simple_minmax'
        quant_loss = quantize_pipeline(
            quant_cfg, net,
            quant_precision=16, retrain=1, verbose=False
        )
    elif q_mode == 'Q21':
        print('AdaQuantE 8bit quantization')
        quant_cfg.ib_quant_method = 'omse'
        quant_loss = quantize_pipeline(
            quant_cfg, net,
            quant_precision=8, retrain=1, verbose=False,
            retrain_iter_num=quant_cfg.retrain_iter_num
        )
    elif q_mode == 'Q22' or q_mode == 'Q2':
        auto_pipeline(quant_cfg, net)

    elif q_mode == 'Q23':
        if not hasattr(quant_cfg, 'mixed_precisions'):
            quant_cfg.mixed_precisions = [4, 8]
        print('MPQ', quant_cfg.mixed_precisions, 'quantization')
        quant_cfg.ib_quant_method = 'omse'
        sim_params = run_mpq(
            quant_cfg, net,
            mp_rate=quant_cfg.mp_rate,
            retrain=1
        )
        quant_cfg.sim_params_file = sim_params
        quant_cfg.torch_params_file = sim_params.replace('simx_', 'torchx_')
    elif q_mode == 'Q30':
        print('Quant aware training(16bit)!')
        setattr(args, 'quant_precision', 16)
        setattr(args, 'finetune', 0)
        if quant_cfg.device:
            setattr(args, 'device', str(quant_cfg.device))
        else:
            setattr(args, 'device', None)
        quant_cfg = run_qat(args, net)
    elif q_mode == 'Q31' or q_mode == 'Q3':
        print('Quant aware training(Default 8bit)!')
        setattr(args, 'quant_precision', 8)
        setattr(args, 'finetune', 0)

        if quant_cfg.device:
            setattr(args, 'device', str(quant_cfg.device))
        else:
            setattr(args, 'device', None)
        quant_cfg = run_qat(args, net)
    elif q_mode == 'Q32':
        # setattr(args, 'quant_precision', 8)
        setattr(args, 'finetune', 1)
        if quant_cfg.device:
            setattr(args, 'device', str(quant_cfg.device))
        else:
            setattr(args, 'device', None)
        assert args.torch_q_param is not None, "[torch_q_param] must be specified in Q32 mode!"
        quant_cfg = run_qat(args, net)
    else:
        pass
    return quant_cfg

def torch_calibrator():
    args = arg_parse()
    model_path = args.model
    image_path = args.image
    if ':' in image_path:
        dir_name = image_path.split(':')[0]
        base_name = image_path.split(':')[-1]
    else:
        if os.path.isdir(image_path):
            dir_name = image_path
            base_name = None
        elif os.path.isfile(image_path):
            dir_name = os.path.abspath(os.path.dirname(image_path))
            base_name = image_path
        else:
            dir_name = image_path
            base_name = None
    input_config = args.input_config
    quant_level = args.quant_level
    q_mode = args.q_mode.upper()
    preprocess = args.preprocess
    num_subsets = args.num_process
    output = args.output
    work_mode = None
    if calibrator_custom.utils.VERSION[:2] in ['S6'] and args.work_mode is not None:
        work_mode = args.work_mode

    if not os.path.exists(model_path):
        raise FileNotFoundError('No such {} model'.format(model_path))

    if not os.path.exists(dir_name):
        raise FileNotFoundError('No such {} image or directory.'.format(image_path))

    quant_cfg = get_quant_config(
        model_path,
        quant_config=args.quant_config,
        input_config=input_config,
        python_file=preprocess,
        calset_dir=dir_name,
        image_list=base_name,
        cal_batchsize=args.cal_batchsize
    )

    net = Net(model_path=model_path, input_config= quant_cfg.cfg_file,
              core_mode=work_mode)

    quant_cfg = set_quant_config(q_mode,quant_cfg,net,args)

    if args.local_rank in [0, -1]:
        net.get_fixed_model(quant_cfg,
                            output_name=output,
                            num_process=num_subsets,
                            quant_level=quant_level)


def arg_parse():
    parser = argparse.ArgumentParser(description='Calibrator Tool')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='Image / Directory containing images path / Image_list for multi_input model.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='float.sim model path.')
    parser.add_argument('--quant_config', type=str, default=None,
                        help='Quant config(yaml) path.')
    parser.add_argument('--input_config', type=str, default=None,
                        help='Input config(ini) path.')
    parser.add_argument('-c', '--category', type=str, default='Unknown',
                        choices=['Classification', 'Detection', 'Unknown'],
                        help='Indicate net category.')
    parser.add_argument('--quant_level', type=str, default='L5',
                        choices=['L1', 'L2', 'L3', 'L4', 'L5'],
                        help='Indicate Quantilization level. The higher the level, the slower the speed and the higher the accuracy.')
    parser.add_argument('-q', '--q_mode', type=str, default='Q0',
                        help='Set Quantization mode')
    parser.add_argument('--q_param', type=str, default=None,
                        help='Set param for specific q_mode')
    # parser.add_argument('--quant_file', type=str, default=None,
    #                     help='Save path for quant_params')
    parser.add_argument('-n', '--preprocess', type=str, default=None,
                        help='Name of model to select image preprocess method')
    parser.add_argument('--num_process', default=10, type=int, help='Amount of processes run at same time.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output path for fixed model.')
    parser.add_argument('--cal_batchsize', type=int, default=100)

    # for quant aware training
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--torch_q_param', type=str, default=None,
                        help='(QAT) torch_params.pkl file for finetune. ')
    parser.add_argument('--multi_gpu', type=int, default=0, help='(QAT) multi_gpu or not')
    parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='(QAT) Select which gpu can be seen')
    parser.add_argument('--local_rank', type=int, default=-1, help='(QAT) processing rank for distributed training')

    if calibrator_custom.utils.VERSION[:2] in ['S6']:
        parser.add_argument('--work_mode', type=str, default=None,
                            choices=['single_core', 'multi_core'],
                            help='Indicate calibrator work_mode.')

    return parser.parse_args()


if __name__ == '__main__':
    torch_calibrator()
