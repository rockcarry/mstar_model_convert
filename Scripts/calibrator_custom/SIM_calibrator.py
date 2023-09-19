# -*- utf-8 -*-

import calibrator_custom
import os
import gc
import sys
import copy
import shutil
import string
import pickle
import numpy as np
import itertools
import configparser
import calibrator_custom
from calibrator_custom import ipu_quantization_lib
from joblib import Parallel, delayed
from collections import OrderedDict
from collections.abc import Iterator
from functools import reduce


STH = 0.002
TARGET_TYPE = 'Fixed'


class SIM_Calibrator(object):
    def __init__(self):
        self._calibrators = OrderedDict()
        self._calibrator_models = OrderedDict()
        self._calibrator_input_configs = OrderedDict()
        self._calibrator_types = OrderedDict()
        self._calibrator_minmax = OrderedDict()
        self._calibrator_chn = OrderedDict()
        self._calibrator_intermediate = OrderedDict()
        self._calibrator_statistics = OrderedDict()
        self._calibrator_qab = OrderedDict()
        self._run_phase = 'R1'
        self._model_inputs = OrderedDict()
        self._model_outputs = OrderedDict()
        self.workers = 1
        self._show_process = None
        self.save_quant_info = False

    def forward(self, *input_data):
        raise NotImplementedError

    def register_calibrator(self, name: str, value) -> None:
        if '_calibrators' not in self.__dict__:
            raise AttributeError(
                "cannot assign calibrators before SIM_Calibrator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("calibrators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("calibrators name can't be empty string \"\"")
        self._calibrators[name] = value
        self._calibrator_models[name] = value.model
        self._calibrator_input_configs[name] = value.input_config
        self._calibrator_types[name] = value.model_type

    def register_model_inputs(self, name: str, value: list) -> None:
        if '_model_inputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_inputs before SIM_Calibrator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("calibrators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("calibrators name can't be empty string \"\"")
        self._model_inputs[name] = value

    def register_model_outputs(self, name: str, value: list) -> None:
        if '_model_outputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_outputs before SIM_Calibrator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("calibrators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("calibrators name can't be empty string \"\"")
        self._model_outputs[name] = value

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_calibrators' not in self.__dict__:
            self._calibrators = OrderedDict()
        if '_model_inputs' not in self.__dict__:
            self._model_inputs = OrderedDict()
        if '_model_outputs' not in self.__dict__:
            self._model_outputs = OrderedDict()
        for name in self._calibrator_models:
            if self._run_phase == 'R1':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_models[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('minmax')
            elif self._run_phase == 'R2':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_models[name], self._calibrator_input_configs[name])
                self._calibrators[name].set_minmax(self._calibrator_minmax[name])
                self._calibrators[name].set_intermediate(self._calibrator_intermediate[name])
                self._set_statistics_mode('statistics')
            elif self._run_phase == 'R3':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_models[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('qab')
            else:
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_models[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('skip')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_calibrators']
        gc.collect()
        return state

    def __setattr__(self, name, value) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        calibrators = self.__dict__.get('_calibrators')
        if calibrator_custom.utils.get_sdk_version() in ['1', 'Q_0']:
            calibrator = sys.modules.get('calibrator_custom._calibrator', False)
        else:
            calibrator = sys.modules.get('calibrator_custom.py_wrapper', False)
        if calibrator and isinstance(value, calibrator.calibrator):
            if value is None:
                raise AttributeError(
                    "cannot assign calibrators before SIM_Calibrator.__init__() call")
                remove_from(self.__dict__, self._calibrators, self._model_inputs, self._model_outputs)
            self.register_calibrator(name, value)
            self.register_model_inputs(name, value.get_input_details())
            self.register_model_outputs(name, value.get_output_details())
        elif calibrators is not None and name in calibrators:
            if value is not None:
                raise TypeError("cannot assign '{}' as calibrators '{}' "
                                "(calibrator_custom.calibrator or None expected)"
                                .format(type(value), name))
            self.register_calibrator(name, value)
            self.register_model_inputs(name, value.get_input_details())
            self.register_model_outputs(name, value.get_output_details())
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_calibrators' in self.__dict__:
            calibrators = self.__dict__['_calibrators']
            if name in calibrators:
                return calibrators[name]
            raise ValueError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._calibrators:
            del self._calibrators[name]
        else:
            object.__delattr__(self, name)

    @staticmethod
    def print_model(model):
        def _get_str(item):
            if isinstance(item, np.ndarray):
                return '[{}]'.format(', '.join(str(v) for v in list(item.flat)))
            else:
                return str(item)

        main_str = '(\n'
        if calibrator_custom.utils.get_sdk_version() in ['S6']:
            main_str += 'work_mode:\t' + model.work_mode + '\n'
        model_inputs = model.get_input_details()
        for num, item in enumerate(model_inputs):
            main_str += 'Input({}):\n'.format(num)
            for key in item:
                main_str += '    ' + key + ':\t' + _get_str(item[key]) + '\n'
        model_outputs = model.get_output_details()
        for num, item in enumerate(model_outputs):
            main_str += 'Output({}):\n'.format(num)
            for key in item:
                main_str += '    ' + key + ':\t' + _get_str(item[key]) + '\n'
        main_str += ')\n'

        return main_str

    def __repr__(self):
        main_str = self.__class__.__name__ + ':\n'
        for name in self._calibrators:
            main_str += name + ' ' + self.print_model(self._calibrators[name])

        return main_str

    def _set_statistics_mode(self, mode):
        for name in self._calibrators:
            self._calibrators[name].set_statistics_mode(mode)

    def _is_data_in_quant_param(self, quant_param):
        if quant_param is None:
            return False
        else:
            for iquant in quant_param:
                if 'data' in iquant:
                    return True
            return False

    def _update_data(self, quant_param=None):
        for name in self._calibrators:
            if self._is_data_in_quant_param(quant_param):
                if (isinstance(self._calibrator_models[name], str)):
                    new_model_path = os.path.join(os.path.dirname(self._calibrator_models[name]),
                        'Origin_' + os.path.basename(self._calibrator_models[name]))
                    print('[INFO] Model ({}) has changed because import quant paramters, origin model has copy to ({}).'.format(
                        self._calibrator_models[name], new_model_path))
                    shutil.copy(self._calibrator_models[name], new_model_path)
                    self._calibrators[name].update_data(quant_param, self._calibrator_models[name])
                else:
                    self._calibrator_models[name] = self._calibrators[name].update_data(quant_param)

    def _check_input_config_all_int16(self):
        checks = []
        for name in self._calibrators:
            conf = configparser.ConfigParser(allow_no_value=True)
            conf.read(self._calibrator_input_configs[name], encoding='utf-8')
            if conf.has_option('CONV_CONFIG', 'input_format') and \
                conf.get('CONV_CONFIG', 'input_format').strip(string.punctuation) == 'ALL_INT16':
                checks.append(True)
            else:
                checks.append(False)
        return all(checks)

    def _check_show_log(self):
        show_log = True
        for name in self._calibrators:
            show_log = show_log and self._calibrators[name].show_log
        with open('/proc/{}/cmdline'.format(os.getppid()), 'r') as fr:
            cmdline = fr.read()
        is_gdb = 'gdb' in cmdline
        self.workers = 1 if is_gdb else self.workers
        with open('/proc/{}/cmdline'.format(os.getpid()), 'r') as fr:
            cmdline = fr.read()
        is_pdb = 'pdb' in cmdline
        return show_log or is_gdb or is_pdb

    def _quantize_models(self, quant_level='L5', quantize_strategy='sp'):
        use_qio = None
        use_qkp = False
        use_qki = False
        if quant_level == 'L2':
            use_qkp = True
        elif quant_level == 'L3':
            use_qio = 'qio'
            use_qkp = True
        elif quant_level == 'L4':
            use_qkp = True
            use_qki = True
        elif quant_level == 'L5':
            use_qio = 'qio'
            use_qkp = True
            use_qki = True

        if quant_level in ['L3', 'L4', 'L5']:
            for name in self._calibrators:
                weights = self._calibrators[name].dump_weights()
                statistics = self._calibrators[name].dump_statistics()
                quant_info_list = ipu_quantization_lib.quantize(weight_tensors_dict=weights, tensor_statistics_dict=statistics, by_channel=use_qkp,
                                  quantize_method=use_qio, suggest_type=use_qki, verbose=False, model=self._calibrators[name],
                                  quantize_strategy=quantize_strategy, parallel=self.workers, sth=STH)
                if self.save_quant_info:
                    quant_file_name = 'quant_info_{}.pkl'.format(name)
                    with open(quant_file_name, 'wb') as f:
                        pickle.dump(quant_info_list, f)
                    print('\nModel {} quant info saved in {}'.format(name, quant_file_name))
                self._calibrators[name].update(quant_info_list)
        else:
            for name in self._calibrators:
                weights = self._calibrators[name].dump_weights()
                quant_info_list = ipu_quantization_lib.quantize(weight_tensors_dict=weights, by_channel=use_qkp, quantize_method=use_qio,
                                  suggest_type=use_qki, verbose=False, model=self._calibrators[name],
                                  quantize_strategy=quantize_strategy, parallel=self.workers, sth=STH)
                if self.save_quant_info:
                    quant_file_name = 'quant_info_{}.pkl'.format(name)
                    with open(quant_file_name, 'wb') as f:
                        pickle.dump(quant_info_list, f)
                    print('\nModel {} quant info saved in {}'.format(name, quant_file_name))
                self._calibrators[name].update(quant_info_list)

    def _convert_model(self, phase, quant_param=None, cmd_model=None, fix_model=None):
        compiler_config = calibrator_custom.utils.CompilerConfig()
        if phase == 'Cmodel_float_with_verify':
            for num, name in enumerate(self._calibrators):
                if self._calibrator_types[name] == 'Float':
                    if cmd_model is None:
                        self._calibrator_models[name] = self._calibrators[name].convert_cmodel_float(compiler_config.Float2CmodelFloatVerifyConfig())
                    else:
                        self._calibrators[name].convert_cmodel_float(compiler_config.Float2CmodelFloatVerifyConfig(), saved_path=cmd_model[num])
                        self._calibrator_models[name] = cmd_model[num]
                    self._calibrator_types[name] = self._calibrators[name].model_type
        elif phase == 'Cmodel_float':
            for num, name in enumerate(self._calibrators):
                if self._calibrator_types[name] == 'Float':
                    if cmd_model is None:
                        if quant_param is not None:
                            self._calibrators[name].update(quant_param)
                        self._calibrator_models[name] = self._calibrators[name].convert_cmodel_float(
                            compiler_config.Float2CmodelFloatConfig())
                    else:
                        if quant_param is not None:
                            self._calibrators[name].update(quant_param)
                        self._calibrators[name].convert_cmodel_float(compiler_config.Float2CmodelFloatConfig(),
                                                                      cmd_model[num])
                        self._calibrator_models[name] = cmd_model[num]
                    self._calibrator_types[name] = self._calibrators[name].model_type
        elif phase == 'OPTIM':
            for num, name in enumerate(self._calibrators):
                if cmd_model is None:
                    self._calibrator_models[name] = self._calibrators[name].convert_cmodel_float(compiler_config.MixedOptimConfig())
                else:
                    self._calibrators[name].convert_cmodel_float(compiler_config.MixedOptimConfig(), cmd_model[num])
                    self._calibrator_models[name] = cmd_model[num]
        elif phase == 'Fixed':
            for num, name in enumerate(self._calibrators):
                if self._calibrator_types[name] == 'Cmodel_float' and TARGET_TYPE == 'Fixed':
                    CompilerPass = compiler_config.CmodelFloat2FixedConfig()
                elif self._calibrator_types[name] == 'Cmodel_float' and TARGET_TYPE == 'Fixed_without_ipu_ctrl':
                    if calibrator_custom.utils.get_sdk_version() not in ['1', 'Q_0']:
                        CompilerPass = compiler_config.CmodelFloat2FixedWOConfig()
                    else:
                        CompilerPass = compiler_config.CmodelFloat2FixedConfig()
                elif self._calibrator_types[name] == 'Fixed_without_ipu_ctrl' and TARGET_TYPE == 'Fixed' and \
                    calibrator_custom.utils.get_sdk_version() not in ['1', 'Q_0']:
                    CompilerPass = compiler_config.FixedWO2Fixed()
                else:
                    raise ValueError('Model type is {}, can not convert to {}.'.format(
                        self._calibrator_types[name], TARGET_TYPE))
                if fix_model is None:
                    self._calibrator_models[name] = self._calibrators[name].convert_fixed(CompilerPass, model_desc=TARGET_TYPE)
                else:
                    self._calibrators[name].convert_fixed(CompilerPass, fix_model[num], TARGET_TYPE)
                    self._calibrator_models[name] = fix_model[num]
                self._calibrator_types[name] = TARGET_TYPE
        else:
            raise ValueError('Error in convert model parameters')

    def _set_param(self, quant_level):
        for name in self._calibrators:
            self._calibrators[name].set_minmax(self._calibrator_minmax[name])
            self._calibrators[name].set_intermediate(self._calibrator_intermediate[name])
            self._calibrators[name].set_chn(self._calibrator_chn[name])
            if quant_level in ['L3', 'L4', 'L5']:
                self._calibrators[name].set_statistics(self._calibrator_statistics[name])

    def _forward_param(self, input_data):
        result_param = {}
        if self._run_phase == 'R1':
            self._set_statistics_mode('minmax')
            self.forward(*input_data)
            for name in self._calibrators:
                result = {}
                result['minmax'] = self._calibrators[name].get_minmax()
                result['intermediate'] = self._calibrators[name].get_intermediate()
                result['chn'] = self._calibrators[name].get_chn()
                result_param[name] = result
        elif self._run_phase == 'R2':
            self._set_statistics_mode('statistics')
            self.forward(*input_data)
            for name in self._calibrators:
                result = {}
                result['statistics'] = self._calibrators[name].get_statistics()
                result_param[name] = result
        else:
            raise ValueError('Error in run model parameters')
        return result_param

    def forward_all(self, inputs):
        r = Parallel(n_jobs=self.workers, backend='multiprocessing')(delayed(self._forward_param)(input_data) for input_data in inputs)
        return r

    def forward_skip(self, inputs):
        r = Parallel(n_jobs=self.workers, backend='multiprocessing')(delayed(self.forward)(*input_data) for input_data in inputs)
        return r

    def _merge_param(self, result_param):
        if self._run_phase == 'R1':
            for name in self._calibrators:
                self._calibrator_minmax[name] = result_param[-1][name]['minmax']
                self._calibrator_intermediate[name] = result_param[-1][name]['intermediate']
                self._calibrator_chn[name] = result_param[-1][name]['chn']
            for result in result_param:
                for name in result:
                    for idx, minmax in enumerate(result[name]['minmax']):
                        assert self._calibrator_minmax[name][idx]['name'] == minmax['name']
                        self._calibrator_minmax[name][idx]['min'] = min(self._calibrator_minmax[name][idx]['min'], minmax['min'])
                        self._calibrator_minmax[name][idx]['max'] = max(self._calibrator_minmax[name][idx]['max'], minmax['max'])
                    for idx, im in enumerate(result[name]['intermediate']):
                        assert self._calibrator_intermediate[name][idx]['op_index'] == im['op_index'] and \
                            self._calibrator_intermediate[name][idx]['in0_name'] == im['in0_name']
                        for jdx, im_minmax in enumerate(im['intermediate_range']):
                            assert self._calibrator_intermediate[name][idx]['intermediate_range'][jdx]['name'] == im_minmax['name']
                            self._calibrator_intermediate[name][idx]['intermediate_range'][jdx]['min'] = list(np.vstack(
                                [np.array(self._calibrator_intermediate[name][idx]['intermediate_range'][jdx]['min']), np.array(im_minmax['min'])]).min(0).flat)
                            self._calibrator_intermediate[name][idx]['intermediate_range'][jdx]['max'] = list(np.vstack(
                                [np.array(self._calibrator_intermediate[name][idx]['intermediate_range'][jdx]['max']), np.array(im_minmax['max'])]).max(0).flat)
                    for idx, chn in enumerate(result[name]['chn']):
                        assert self._calibrator_chn[name][idx]['name'] == chn['name']
                        self._calibrator_chn[name][idx]['min'] = list(np.vstack(
                            [np.array(self._calibrator_chn[name][idx]['min']), np.array(chn['min'])]).min(0).flat)
                        self._calibrator_chn[name][idx]['max'] = list(np.vstack(
                            [np.array(self._calibrator_chn[name][idx]['max']), np.array(chn['max'])]).max(0).flat)
        elif self._run_phase == 'R2':
            for name in self._calibrators:
                self._calibrator_statistics[name] = result_param[-1][name]['statistics']
            for result in result_param[:-1]:
                for name in result:
                    for idx, statistics in enumerate(result[name]['statistics']):
                        assert self._calibrator_statistics[name][idx]['name'] == statistics['name']
                        self._calibrator_statistics[name][idx]['bin'] = np.add(self._calibrator_statistics[name][idx]['bin'], statistics['bin'])
        else:
            raise ValueError('Error in run model parameters')

    def _convert_import_impl(self, quant_param, cmd_model=None, fix_model=None):
        self._run_phase = 'R3'
        self._convert_model('Cmodel_float', quant_param=quant_param, cmd_model=cmd_model)
        self._convert_model('Fixed', fix_model=fix_model)

    def __show_and_feed_input(self, gen):
        for g in gen:
            if not self._check_show_log():
                self._show_process.show_process()
            yield g

    def _convert_impl(self, inputs, quant_level='L5', mixed_precision='sp', quant_param=None,
                      cmd_model=None, fix_model=None):
        gen_minmax, gen_statistics, gen_nums = itertools.tee(*inputs, 3)
        num_images = len(list(gen_nums))
        show_process_len = num_images * 2 if quant_level in ['L3', 'L4', 'L5'] else num_images
        self.workers = num_images if num_images < self.workers else self.workers
        self._check_show_log()
        self._show_process = calibrator_custom.utils.ShowProcess(show_process_len)
        self._update_data(quant_param)
        self._merge_param(self.forward_all(self.__show_and_feed_input(gen_minmax)))
        self._run_phase = 'R2'
        if quant_level in ['L3', 'L4', 'L5']:
            self._merge_param(self.forward_all(self.__show_and_feed_input(gen_statistics)))
        if not self._check_show_log():
            self._show_process.start_show_convert()
        self._set_param(quant_level)
        self._quantize_models(quant_level, mixed_precision)
        self._run_phase = 'R3'
        self._convert_model('Cmodel_float', quant_param=quant_param, cmd_model=cmd_model)
        self._convert_model('Fixed', fix_model=fix_model)
        self._show_process.close_show_convert()

    def convert(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            self.workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            self.workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        quant_level = kwargs['quant_level'] if 'quant_level' in kwargs else 'L5'
        quant_param = kwargs['quant_param'] if 'quant_param' in kwargs else None
        mixed_precision = kwargs['mixed_precision'] if 'mixed_precision' in kwargs else 'sp'
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        fix_model = kwargs['fix_model'] if 'fix_model' in kwargs else None

        if quant_level not in ['L1', 'L2', 'L3', 'L4', 'L5']:
            raise ValueError('Unknown quant_level: {}'.format(quant_level))
        calibrator_custom.utils.check_quant_param(quant_param)
        if self._check_input_config_all_int16() and quant_level != 'L1':
            quant_level = 'L2'
        if quant_param is not None:
            quant_level = 'L1'

        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if fix_model is not None and len(fix_model) != len(self._calibrators):
            raise ValueError('Num of fix_models must equal Num of models')
        cmodel_models = []
        for name in self._calibrators:
            if isinstance(self._calibrator_models[name], str):
                cmodel_models.append(calibrator_custom.utils.get_out_model_name(self._calibrator_models[name], phase='CMD_Float'))
        if cmd_model is None and len(cmodel_models) > 0:
            cmd_model = cmodel_models

        if inputs == ():
            if quant_param is None:
                raise ValueError('quant_param cannot be None')
            self._convert_import_impl(quant_param, cmd_model=cmd_model, fix_model=fix_model)
        else:
            if not isinstance(*inputs, Iterator):
                raise ValueError('SIM_Calibrator need data generator for input')
            self._convert_impl(inputs, quant_level=quant_level, mixed_precision=mixed_precision,
                        quant_param=quant_param, cmd_model=cmd_model, fix_model=fix_model)

        if fix_model is None:
            return list(self._calibrator_models.values())

    def convert_fixed(self, **kwargs):
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        fix_model = kwargs['fix_model'] if 'fix_model' in kwargs else None

        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if fix_model is not None and len(fix_model) != len(self._calibrators):
            raise ValueError('Num of fix_models must equal Num of models')
        cmodel_models = []
        for name in self._calibrators:
            if isinstance(self._calibrator_models[name], str):
                cmodel_models.append(calibrator_custom.utils.get_out_model_name(self._calibrator_models[name], phase='CMD_Float'))
        if cmd_model is None and len(cmodel_models) > 0:
            cmd_model = cmodel_models

        self._convert_model('Cmodel_float_with_verify', cmd_model=cmd_model)
        self._convert_model('Fixed', fix_model=fix_model)

    def _convert_all_int16(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            self.workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            self.workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        cmodel_models = []
        for name in self._calibrators:
            if isinstance(self._calibrator_models[name], str):
                cmodel_models.append(calibrator_custom.utils.get_out_model_name(self._calibrator_models[name], phase='CMD_Float'))
        if cmd_model is None and len(cmodel_models) > 0:
            cmd_model = cmodel_models

        tensor_details = OrderedDict()
        conv_details = OrderedDict()
        quant_params = []
        gen_minmax, gen_minmax_one = itertools.tee(*inputs, 2)
        self._run_phase = 'R1'
        self._merge_param(self.forward_all(gen_minmax))
        self._set_param('L1')
        self._convert_model('OPTIM', cmd_model=cmd_model)
        for name in self._calibrators:
            tensor_details[name] = self._calibrators[name].get_tensor_details()
            conv_details[name] = self._calibrators[name].get_op_details(['CONV_2D', 'DEPTHWISE_CONV_2D'])
        for name in self._calibrators:
            quant_info = OrderedDict()
            for name in conv_details:
                for conv in conv_details[name]:
                    if 16 in conv['ib_allowed_bits']:
                        quant_info['name'] = conv['input_tensors'][0]
                        quant_info['min'] = conv['input_min']
                        quant_info['max'] = conv['input_max']
                        quant_info['bit'] = 16
                        quant_params.append(copy.deepcopy(quant_info))
                        quant_info['name'] = conv['input_tensors'][1]
                        quant_info['min'] = conv['weights_min']
                        quant_info['max'] = conv['weights_max']
                        quant_info['bit'] = 16
                        quant_params.append(copy.deepcopy(quant_info))
                self._convert_model('OPTIM', quant_param=quant_params, cmd_model=cmd_model)
        return quant_params

    def convert_optim_model(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            self.workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            self.workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        cmodel_models = []
        for name in self._calibrators:
            if isinstance(self._calibrator_models[name], str):
                cmodel_models.append(calibrator_custom.utils.get_out_model_name(self._calibrator_models[name], phase='CMD_Float'))
        if cmd_model is None and len(cmodel_models) > 0:
            cmd_model = cmodel_models

        conv_details = OrderedDict()
        conv_params = []
        gen_minmax, gen_statistics, gen_test1, gen_test2 = itertools.tee(*inputs, 4)
        self._run_phase = 'R1'
        self._merge_param(self.forward_all(gen_minmax))
        self._run_phase = 'R2'
        self._merge_param(self.forward_all(gen_statistics))
        self._set_param('L5')
        self._quantize_models('L5')
        self._convert_model('OPTIM', cmd_model=cmd_model)
        for name in self._calibrators:
            conv_details[name] = self._calibrators[name].get_op_details(['CONV_2D', 'DEPTHWISE_CONV_2D'])
        for name in self._calibrators:
            for conv in conv_details[name]:
                conv_params.append(conv)
        return conv_params

    def _optimize_run(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            self.workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            self.workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        quant_param = kwargs['quant_param']
        calibrator_custom.utils.check_quant_param(quant_param)
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        cmodel_models = []
        for name in self._calibrators:
            if isinstance(self._calibrator_models[name], str):
                cmodel_models.append(calibrator_custom.utils.get_out_model_name(self._calibrator_models[name], phase='CMD_Float'))
        if cmd_model is None and len(cmodel_models) > 0:
            cmd_model = cmodel_models

        self._convert_model('OPTIM', quant_param=quant_param, cmd_model=cmd_model)
        self._run_phase = 'RUN'
        return self.forward_skip(*inputs)

    def optimize(self, inputs, conv8_info, float_result, compression_rates, num_subsets=10, conv16_chn=True):
        gen_int16, inputs = itertools.tee(inputs, 2)
        conv16_minmax = self._convert_all_int16(gen_int16, num_process=num_subsets)
        if conv16_chn:
            for index, conv8 in enumerate(conv8_info):
                for conv16 in conv16_minmax:
                    if conv16['name'] == conv8['input_tensors'][1]:
                        conv16['min'] = conv8['weights_min']
                        conv16['max'] = conv8['weights_max']
        convs_info = []
        img_gens = itertools.tee(inputs, len(conv8_info))
        for index, conv8 in enumerate(conv8_info):
            quant_info = copy.deepcopy(conv16_minmax)
            for conv16 in quant_info:
                if conv16['name'] == conv8['input_tensors'][0] and 8 in conv8['ib_allowed_bits']:
                    conv16['min'] = conv8['input_min']
                    conv16['max'] = conv8['input_max']
                    conv16['bit'] = 8
                if conv16['name'] == conv8['input_tensors'][1] and 8 in conv8['ib_allowed_bits']:
                    conv16['min'] = conv8['weights_min']
                    conv16['max'] = conv8['weights_max']
                    conv16['bit'] = 8
            int_result = self._optimize_run(img_gens[index], quant_param=quant_info, num_process=num_subsets)
            conv_info_dict = OrderedDict()
            conv_info_dict['name'] = str(conv8['op_index'])
            conv_info_dict['ib_size'] = reduce(lambda x, y: x * y, conv8['input_shape'])
            conv_info_dict['kb_size'] = reduce(lambda x, y: x * y, conv8['weights_shape'])
            convs_info.append(conv_info_dict)
            convs_info = ipu_quantization_lib.quantize_mp_8_16(convs_info, output_sample=int_result, output_benchmark=float_result)
        convs_info = ipu_quantization_lib.quantize_mp_8_16(convs_info, compression_rates=compression_rates)

        optimized_quant_infos = []
        for idx, rate in enumerate(compression_rates):
            quant_info = copy.deepcopy(conv16_minmax)
            for sen_conv in convs_info:
                for index, conv8 in enumerate(conv8_info):
                    for conv16 in quant_info:
                        if sen_conv['name'] == str(conv8['op_index']) and conv16['name'] == conv8['input_tensors'][0]:
                            if sen_conv['quant_precision'][idx] == 8:
                                conv16['min'] = conv8['input_min']
                                conv16['max'] = conv8['input_max']
                                conv16['bit'] = 8
                        if sen_conv['name'] == str(conv8['op_index']) and conv16['name'] == conv8['input_tensors'][1]:
                            if sen_conv['quant_precision'][idx] == 8:
                                conv16['min'] = conv8['weights_min']
                                conv16['max'] = conv8['weights_max']
                                conv16['bit'] = 8
            optimized_quant_infos.append(quant_info)

        return optimized_quant_infos
