# -*- utf-8 -*-

import calibrator_custom
import os
import numpy as np
import platform
from joblib import Parallel, delayed
from collections import OrderedDict
from collections.abc import Iterator


class SIM_Simulator(object):
    def __init__(self):
        self._simulators = OrderedDict()
        self._simulator_paths = OrderedDict()
        self._simulator_class_types = OrderedDict()
        self._model_inputs = OrderedDict()
        self._model_outputs = OrderedDict()

    def forward(self, *input_data):
        raise NotImplementedError

    def register_simulator(self, name: str, value) -> None:
        if '_simulators' not in self.__dict__:
            raise AttributeError(
                "cannot assign simulators before SIM_Simulator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("simulators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("simulators name can't be empty string \"\"")
        self._simulators[name] = value
        self._simulator_paths[name] = value.model_path
        self._simulator_class_types[name] = value.class_type

    def register_model_inputs(self, name: str, value: list) -> None:
        if '_model_inputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_inputs before SIM_Simulator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("simulators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("simulators name can't be empty string \"\"")
        self._model_inputs[name] = value

    def register_model_outputs(self, name: str, value: list) -> None:
        if '_model_outputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_outputs before SIM_Simulator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("simulators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("simulators name can't be empty string \"\"")
        self._model_outputs[name] = value

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_simulators' not in self.__dict__:
            self._simulators = OrderedDict()
        if '_model_inputs' not in self.__dict__:
            self._model_inputs = OrderedDict()
        if '_model_outputs' not in self.__dict__:
            self._model_outputs = OrderedDict()
        for name in self._simulator_paths:
            if self._simulator_class_types[name] == 'FLOAT':
                self._simulators[name] = calibrator_custom.float_simulator(self._simulator_paths[name])
            elif self._simulator_class_types[name] == 'CMODEL_FLOAT':
                self._simulators[name] = calibrator_custom.cmodel_float_simulator(self._simulator_paths[name])
            elif self._simulator_class_types[name] == 'CMODEL_FIXED_WITHOUT_IPU_CTRL':
                self._simulators[name] = calibrator_custom.fixed_wipu_simulator(self._simulator_paths[name])
            elif self._simulator_class_types[name] == 'FIXED':
                self._simulators[name] = calibrator_custom.fixed_simulator(self._simulator_paths[name])
            elif self._simulator_class_types[name] == 'OFFLINE':
                self._simulators[name] = calibrator_custom.offline_simulator(self._simulator_paths[name])
            else:
                raise KeyError("Unknown simulator class_type.")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_simulators']
        return state

    def __setattr__(self, name, value) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        simulators = self.__dict__.get('_simulators')
        if platform.architecture()[0][:2] == '64':
            if isinstance(value, calibrator_custom.float_simulator):
                if value is None:
                    raise AttributeError(
                        "cannot assign simulators before SIM_Simulator.__init__() call")
                    remove_from(self.__dict__, self._simulators, self._model_inputs, self._model_outputs)
                self.register_simulator(name, value)
                self.register_model_inputs(name, value.get_input_details())
                self.register_model_outputs(name, value.get_output_details())
            elif simulators is not None and name in simulators:
                if value is not None:
                    raise TypeError("cannot assign '{}' as simulators '{}' "
                                    "(calibrator_custom.simulator or None expected)"
                                    .format(type(value), name))
                self.register_simulator(name, value)
                self.register_model_inputs(name, value.get_input_details())
                self.register_model_outputs(name, value.get_output_details())
            else:
                object.__setattr__(self, name, value)

        elif platform.architecture()[0][:2] == '32':
            if isinstance(value, calibrator_custom.fixed_simulator) or \
                isinstance(value, calibrator_custom.offline_simulator):
                if value is None:
                    raise AttributeError(
                        "cannot assign simulators before SIM_Simulator.__init__() call")
                    remove_from(self.__dict__, self._simulators, self._model_inputs, self._model_outputs)
                self.register_simulator(name, value)
                self.register_model_inputs(name, value.get_input_details())
                self.register_model_outputs(name, value.get_output_details())
            elif simulators is not None and name in simulators:
                if value is not None:
                    raise TypeError("cannot assign '{}' as simulators '{}' "
                                    "(calibrator_custom.simulator or None expected)"
                                    .format(type(value), name))
                self.register_simulator(name, value)
                self.register_model_inputs(name, value.get_input_details())
                self.register_model_outputs(name, value.get_output_details())
            else:
                object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_simulators' in self.__dict__:
            simulators = self.__dict__['_simulators']
            if name in simulators:
                return simulators[name]
            raise ValueError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._simulators:
            del self._simulators[name]
        else:
            object.__delattr__(self, name)

    def _get_str(self, item):
        if isinstance(item, np.ndarray):
            return '[{}]'.format(', '.join(str(v) for v in list(item.flat)))
        else:
            return str(item)

    def __repr__(self):
        main_str = self.__class__.__name__ + ':\n'
        for name in self._simulators:
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

    def forward_all(self, inputs, workers):
        r = Parallel(n_jobs=workers)(delayed(self.forward)(*input_data) for input_data in inputs)
        return r

    def _call_impl(self, *inputs, **kwargs):
        if 'num_process' in kwargs:
            workers = kwargs['num_process'] if kwargs['num_process'] < os.cpu_count() else os.cpu_count()
            if not isinstance(*inputs, Iterator) and workers > 1:
                raise ValueError('Multi process mode need data generator for input')
            elif isinstance(*inputs, Iterator):
                result = self.forward_all(*inputs, workers)
            else:
                result = self.forward(*inputs)
        else:
            result = self.forward(*inputs, **kwargs)

        return result

    def __call__(self, *inputs, **kwargs):
        result = self._call_impl(*inputs, **kwargs)
        return result




