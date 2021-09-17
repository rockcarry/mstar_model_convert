# -*- utf-8 -*-

import calibrator_custom
import os
import copy
import numpy as np
import itertools
from calibrator_custom import quantize
from calibrator_custom import utils
from joblib import Parallel, delayed
from collections import OrderedDict
from collections.abc import Iterator
from functools import reduce


class SIM_Calibrator(object):
    def __init__(self):
        self._calibrators = OrderedDict()
        self._calibrator_paths = OrderedDict()
        self._calibrator_input_configs = OrderedDict()
        self._calibrator_minmax = OrderedDict()
        self._calibrator_chn = OrderedDict()
        self._calibrator_lut = OrderedDict()
        self._calibrator_statistics = OrderedDict()
        self._calibrator_qab = OrderedDict()
        self._run_phase = 'R1'
        self._model_inputs = OrderedDict()
        self._model_outputs = OrderedDict()

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
        self._calibrator_paths[name] = value.model_path
        self._calibrator_input_configs[name] = value.input_config

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
        for name in self._calibrator_paths:
            if self._run_phase == 'R1':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_paths[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('minmax')
            elif self._run_phase == 'R2':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_paths[name], self._calibrator_input_configs[name])
                self._calibrators[name].set_minmax(self._calibrator_minmax[name])
                self._calibrators[name].set_lut(self._calibrator_lut[name])
                self._set_statistics_mode('statistics')
            elif self._run_phase == 'R3':
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_paths[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('qab')
            else:
                self._calibrators[name] = calibrator_custom.calibrator(self._calibrator_paths[name], self._calibrator_input_configs[name])
                self._set_statistics_mode('skip')

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_calibrators']
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
        if isinstance(value, calibrator_custom.calibrator):
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

    def _get_str(self, item):
        if isinstance(item, np.ndarray):
            return '[{}]'.format(', '.join(str(v) for v in list(item.flat)))
        else:
            return str(item)

    def __repr__(self):
        main_str = self.__class__.__name__ + ':\n'
        for name in self._calibrators:
            main_str += name + ' (\n'
            for num, item in enumerate(self._model_inputs[name]):
                main_str += 'Input({}):\n'.format(num)
                for key in item:
                    main_str += '    ' + key + ':\t' + self._get_str(item[key]) + '\n'
            for num, item in enumerate(self._model_outputs[name]):
                main_str += 'Output({}):\n'.format(num)
                for key in item:
                    main_str += '    ' + key + ':\t' + self._get_str(item[key]) + '\n'
            main_str += ')\n'

        return main_str

    def _set_statistics_mode(self, mode):
        for name in self._calibrators:
            self._calibrators[name].set_statistics_mode(mode)

    def _update_data(self, quant_param=None):
        for name in self._calibrators:
            self._calibrators[name].update_data(quant_param, self._calibrator_paths[name])

    def _quantize_models(self, workers, quant_level='L5', quantize_strategy='sp'):
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
        else:
            return

        for name in self._calibrators:
            weights = self._calibrators[name].dump_weights()
            statistics = self._calibrators[name].dump_statistics()
            quantize.quantize(weight_tensors_dict=weights, tensor_statistics_dict=statistics, by_channel=use_qkp,
                     quantize_method=use_qio, suggest_type=use_qki, verbose=False, model=self._calibrators[name],
                     quantize_strategy=quantize_strategy, parallel=workers, sth=0.002)

    def _convert_model(self, phase, quant_param=None, Dev_config=None, cmd_model=None, fix_model=None):
        infect_datatype_config_file = './infectDataTypeConfig.txt'
        if phase == 'P0':
            with open(infect_datatype_config_file, 'w') as fd:
                fd.write('fuse_elementwise_to_preceding_ops\n')
                fd.write('fuse_activation_functions\n')
                fd.write('process_st_mimmax\n')
                fd.write('datatype_infect_and_verify\n')
                fd.write('quant_domain_infect_and_verify\n')
                fd.write('calcu_qab_minmax\n')
            for num, name in enumerate(self._calibrators):
                self._calibrators[name].p0_convert(infect_datatype_config_file, cmd_model[num])
                self._calibrator_paths[name] = cmd_model[num]
            os.remove(infect_datatype_config_file)
        elif phase == 'P1':
            with open(infect_datatype_config_file, 'w') as fd:
                fd.write('fuse_elementwise_to_preceding_ops\n')
                fd.write('fuse_activation_functions\n')
                fd.write('process_st_mimmax\n')
                fd.write('infect_datatype\n')
                fd.write('infect_quant_domain\n')
                fd.write('calcu_qab_minmax\n')
            for num, name in enumerate(self._calibrators):
                self._calibrators[name].p1_convert(infect_datatype_config_file, quant_param, cmd_model[num])
                self._calibrator_paths[name] = cmd_model[num]
            os.remove(infect_datatype_config_file)
        elif phase == 'OPTIM':
            with open(infect_datatype_config_file, 'w') as fd:
                fd.write('process_st_mimmax\n')
                fd.write('infect_datatype\n')
                fd.write('infect_quant_domain\n')
            for num, name in enumerate(self._calibrators):
                self._calibrators[name].p1_convert(infect_datatype_config_file, quant_param, cmd_model[num])
                self._calibrator_paths[name] = cmd_model[num]
            os.remove(infect_datatype_config_file)
        elif phase == 'P2':
            with open(infect_datatype_config_file, 'w'):
                pass
            for num, name in enumerate(self._calibrators):
                self._calibrators[name].p2_convert(infect_datatype_config_file, cmd_model[num])
            os.remove(infect_datatype_config_file)
        elif phase == 'P3':
            for num, name in enumerate(self._calibrators):
                self._calibrators[name].p3_convert(Dev_config, fix_model[num])
        else:
            raise ValueError('Error in convert model parameters')

    def _forward_one(self, input_gen):
        if self._run_phase == 'R1':
            self._set_statistics_mode('minmax')
            self.forward(*next(input_gen))
            for name in self._calibrators:
                self._calibrators[name].set_minmax(self._calibrator_minmax[name])
                self._calibrators[name].set_lut(self._calibrator_lut[name])
                self._calibrators[name].set_chn(self._calibrator_chn[name])
        elif self._run_phase == 'R2':
            self._set_statistics_mode('statistics')
            for name in self._calibrators:
                self._calibrators[name].set_minmax(self._calibrator_minmax[name])
                self._calibrators[name].set_lut(self._calibrator_lut[name])
            self.forward(*next(input_gen))
            for name in self._calibrators:
                self._calibrators[name].set_statistics(self._calibrator_statistics[name])
        elif self._run_phase == 'R3':
            self._set_statistics_mode('qab')
            self.forward(*next(input_gen))
            for name in self._calibrators:
                self._calibrators[name].set_qab(self._calibrator_qab[name])
                self._calibrators[name].set_lut(self._calibrator_lut[name])
        else:
            raise ValueError('Error in forward model parameters')

    def _forward_param(self, input_data):
        result_param = {}
        if self._run_phase == 'R1':
            self._set_statistics_mode('minmax')
            self.forward(*input_data)
            for name in self._calibrators:
                result = {}
                result['minmax'] = self._calibrators[name].get_minmax()
                result['lut'] = self._calibrators[name].get_lut()
                result['chn'] = self._calibrators[name].get_chn()
                result_param[name] = result
        elif self._run_phase == 'R2':
            self._set_statistics_mode('statistics')
            self.forward(*input_data)
            for name in self._calibrators:
                result = {}
                result['statistics'] = self._calibrators[name].get_statistics()
                result_param[name] = result
        elif self._run_phase == 'R3':
            self._set_statistics_mode('qab')
            self.forward(*input_data)
            for name in self._calibrators:
                result = {}
                result['qab'] = self._calibrators[name].get_qab()
                result['lut'] = self._calibrators[name].get_lut()
                result_param[name] = result
        else:
            raise ValueError('Error in run model parameters')
        return result_param

    def forward_all(self, inputs, workers):
        r = Parallel(n_jobs=workers)(delayed(self._forward_param)(input_data) for input_data in inputs)
        return r

    def forward_skip(self, inputs, workers):
        r = Parallel(n_jobs=workers)(delayed(self.forward)(*input_data) for input_data in inputs)
        return r

    def _merge_param(self, result_param):
        if self._run_phase == 'R1':
            for name in self._calibrators:
                self._calibrator_minmax[name] = result_param[0][name]['minmax']
                self._calibrator_lut[name] = result_param[0][name]['lut']
                self._calibrator_chn[name] = result_param[0][name]['chn']
            for result in result_param:
                for name in result:
                    for idx, minmax in enumerate(result[name]['minmax']):
                        assert self._calibrator_minmax[name][idx]['name'] == minmax['name']
                        self._calibrator_minmax[name][idx]['min'] = min(self._calibrator_minmax[name][idx]['min'], minmax['min'])
                        self._calibrator_minmax[name][idx]['max'] = max(self._calibrator_minmax[name][idx]['max'], minmax['max'])
                    for idx, lut in enumerate(result[name]['lut']):
                        assert self._calibrator_lut[name][idx]['name'] == lut['name']
                        self._calibrator_lut[name][idx]['min'] = min(self._calibrator_lut[name][idx]['min'], lut['min'])
                        self._calibrator_lut[name][idx]['max'] = max(self._calibrator_lut[name][idx]['max'], lut['max'])
                    for idx, chn in enumerate(result[name]['chn']):
                        assert self._calibrator_chn[name][idx]['name'] == chn['name']
                        self._calibrator_chn[name][idx]['min'] = list(np.vstack(
                            [np.array(self._calibrator_chn[name][idx]['min']), np.array(chn['min'])]).min(0).flat)
                        self._calibrator_chn[name][idx]['max'] = list(np.vstack(
                            [np.array(self._calibrator_chn[name][idx]['max']), np.array(chn['max'])]).max(0).flat)
        elif self._run_phase == 'R2':
            for name in self._calibrators:
                self._calibrator_statistics[name] = result_param[0][name]['statistics']
            for result in result_param:
                for name in result:
                    for idx, statistics in enumerate(result[name]['statistics']):
                        assert self._calibrator_statistics[name][idx]['name'] == statistics['name']
                        self._calibrator_statistics[name][idx]['bin'] = np.add(self._calibrator_statistics[name][idx]['bin'], statistics['bin'])
        elif self._run_phase == 'R3':
            for name in self._calibrators:
                self._calibrator_qab[name] = result_param[0][name]['qab']
                self._calibrator_lut[name] = result_param[0][name]['lut']
            for result in result_param:
                for name in result:
                    for idx, qab in enumerate(result[name]['qab']):
                        assert self._calibrator_qab[name][idx]['name'] == qab['name']
                        self._calibrator_qab[name][idx]['min'] = list(np.vstack(
                            [np.array(self._calibrator_qab[name][idx]['min']), np.array(qab['min'])]).min(0).flat)
                        self._calibrator_qab[name][idx]['max'] = list(np.vstack(
                            [np.array(self._calibrator_qab[name][idx]['max']), np.array(qab['max'])]).max(0).flat)
                    for idx, lut in enumerate(result[name]['lut']):
                        assert self._calibrator_lut[name][idx]['name'] == lut['name']
                        self._calibrator_lut[name][idx]['min'] = min(self._calibrator_lut[name][idx]['min'], lut['min'])
                        self._calibrator_lut[name][idx]['max'] = max(self._calibrator_lut[name][idx]['max'], lut['max'])
        else:
            raise ValueError('Error in run model parameters')

    def _convert_impl(self, inputs, workers, quant_level='L5', mixed_precision='sp', quant_param=None, Dev_config=None,
                      cmd_model=None, fix_model=None):
        gen_minmax, gen_statistics, gen_qab, gen_test1, gen_test2, gen_test3 = itertools.tee(*inputs, 6)
        self._update_data(quant_param)
        self._merge_param(self.forward_all(gen_minmax, workers))
        self._forward_one(gen_test1)
        self._run_phase = 'R2'
        if quant_param is None and quant_level != 'L1':
            self._merge_param(self.forward_all(gen_statistics, workers))
            self._forward_one(gen_test2)
            self._quantize_models(workers, quant_level, mixed_precision)
        self._run_phase = 'R3'
        self._convert_model('P1', quant_param=quant_param, cmd_model=cmd_model)
        self._convert_model('P3', Dev_config=Dev_config, fix_model=fix_model)

    def convert(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        quant_level = kwargs['quant_level'] if 'quant_level' in kwargs else 'L5'
        quant_param = kwargs['quant_param'] if 'quant_param' in kwargs else None
        mixed_precision = kwargs['mixed_precision'] if 'mixed_precision' in kwargs else 'sp'
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        fix_model = kwargs['fix_model']

        if quant_level not in ['L1', 'L2', 'L3', 'L4', 'L5']:
            raise ValueError('Unknown quant_level: {}'.format(quant_level))
        utils.check_quant_param(quant_param)
        Dev_config = utils.get_dev_config()

        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if cmd_model is None:
            cmd_model = []
            for name in self._calibrators:
                cmd_model.append(utils.get_out_model_name(self._calibrator_paths[name], phase='CMD_Fload'))

        if len(fix_model) != len(self._calibrators):
            raise ValueError('Num of fix_models must equal Num of models')

        self._convert_impl(inputs, workers, quant_level=quant_level, mixed_precision=mixed_precision,
                    quant_param=quant_param, Dev_config=Dev_config, cmd_model=cmd_model, fix_model=fix_model)

    def convert_fixed(self, **kwargs):
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        fix_model = kwargs['fix_model']
        Dev_config = utils.get_dev_config()

        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if cmd_model is None:
            cmd_model = []
            for name in self._calibrators:
                cmd_model.append(utils.get_out_model_name(self._calibrator_paths[name], phase='CMD_Fload'))

        if len(fix_model) != len(self._calibrators):
            raise ValueError('Num of fix_models must equal Num of models')

        self._convert_model('P0', cmd_model=cmd_model)
        self._convert_model('P3', Dev_config=Dev_config, fix_model=fix_model)

    def _convert_all_int16(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if cmd_model is None:
            cmd_model = []
            for name in self._calibrators:
                cmd_model.append(utils.get_out_model_name(self._calibrator_paths[name], phase='CMD_Fload'))

        tensor_details = OrderedDict()
        conv_details = OrderedDict()
        quant_params = []
        gen_minmax, gen_minmax_one = itertools.tee(*inputs, 2)
        self._run_phase = 'R1'
        self._merge_param(self.forward_all(gen_minmax, workers))
        self._forward_one(gen_minmax_one)
        self._convert_model('OPTIM', cmd_model=cmd_model)
        for name in self._calibrators:
            tensor_details[name] = self._calibrators[name].get_tensor_details()
            conv_details[name] = self._calibrators[name].get_op_details(['Conv2D', 'DepthwiseConv2D'])
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
            workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if cmd_model is None:
            cmd_model = []
            for name in self._calibrators:
                cmd_model.append(utils.get_out_model_name(self._calibrator_paths[name], phase='CMD_Fload'))

        conv_details = OrderedDict()
        conv_params = []
        gen_minmax, gen_statistics, gen_test1, gen_test2 = itertools.tee(*inputs, 4)
        self._run_phase = 'R1'
        self._merge_param(self.forward_all(gen_minmax, workers))
        self._forward_one(gen_test1)
        self._run_phase = 'R2'
        self._merge_param(self.forward_all(gen_statistics, workers))
        self._forward_one(gen_test2)
        self._quantize_models(workers, 'L5')
        self._convert_model('OPTIM', cmd_model=cmd_model)
        for name in self._calibrators:
            conv_details[name] = self._calibrators[name].get_op_details(['Conv2D', 'DepthwiseConv2D'])
        for name in self._calibrators:
            for conv in conv_details[name]:
                conv_params.append(conv)
        return conv_params

    def _optimize_run(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
        else:
            workers = 10 if 10 < os.cpu_count() else os.cpu_count()

        if not isinstance(*inputs, Iterator):
            raise ValueError('SIM_Calibrator need data generator for input')

        quant_param = kwargs['quant_param']
        utils.check_quant_param(quant_param)
        cmd_model = kwargs['cmd_model'] if 'cmd_model' in kwargs else None
        if cmd_model is not None and len(cmd_model) != len(self._calibrators):
            raise ValueError('Num of cmd_models must equal Num of models')
        if cmd_model is None:
            cmd_model = []
            for name in self._calibrators:
                cmd_model.append(self._calibrator_paths[name])

        self._convert_model('OPTIM', quant_param=quant_param, cmd_model=cmd_model)
        self._run_phase = 'RUN'
        return self.forward_skip(*inputs, workers)

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
            convs_info = quantize.quantize_mp_8_16(convs_info, output_sample=int_result, output_benchmark=float_result)
        convs_info = quantize.quantize_mp_8_16(convs_info, compression_rates=compression_rates)

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



