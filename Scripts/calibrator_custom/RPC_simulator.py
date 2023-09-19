# -*- utf-8 -*-

import calibrator_custom
import os
import gc
import sys
import numpy as np
import itertools
from calibrator_custom import utils
from collections import OrderedDict
from collections.abc import Iterator


class RPC_Simulator(object):
    def __init__(self):
        self._simulators = OrderedDict()
        self._model_inputs = OrderedDict()
        self._model_outputs = OrderedDict()
        self._show_process = None

    def forward(self, *input_data):
        raise NotImplementedError

    def register_simulator(self, name: str, value) -> None:
        if '_simulators' not in self.__dict__:
            raise AttributeError(
                "cannot assign simulators before RPC_Simulator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("simulators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("simulators name can't be empty string \"\"")
        self._simulators[name] = value

    def register_model_inputs(self, name: str, value: list) -> None:
        if '_model_inputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_inputs before RPC_Simulator.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("simulators name should be a string. "
                            "Got {}".format(type(name)))
        elif name == '':
            raise KeyError("simulators name can't be empty string \"\"")
        self._model_inputs[name] = value

    def register_model_outputs(self, name: str, value: list) -> None:
        if '_model_outputs' not in self.__dict__:
            raise AttributeError(
                "cannot assign model_outputs before RPC_Simulator.__init__() call")
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_simulators']
        gc.collect()
        return state

    def __setattr__(self, name, value) -> None:
        from calibrator_custom import rpc_simulator
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        simulators = self.__dict__.get('_simulators')
        simulator = sys.modules.get('calibrator_custom.rpc_simulator', False)
        if simulator and isinstance(value, simulator.simulator):
            if value is None:
                raise AttributeError(
                    "cannot assign simulators before RPC_Simulator.__init__() call")
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

    @staticmethod
    def print_model(model):
        def _get_str(item):
            if isinstance(item, np.ndarray):
                return '[{}]'.format(', '.join(str(v) for v in list(item.flat)))
            else:
                return str(item)

        main_str = '(\n'
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
        for name in self._simulators:
            main_str += name + ' ' + self.print_model(self._simulators[name])

        return main_str

    def forward_all(self, inputs):
        r = []
        for input_data in inputs:
            self._show_process.show_process()
            r.append(self.forward(*input_data))
        return r

    def _call_impl(self, *inputs, **kwargs):
        if isinstance(*inputs, Iterator):
            gen_inputs, gen_test = itertools.tee(*inputs, 2)
            self._show_process = utils.ShowProcess(len(list(gen_test)))
            result = self.forward_all(gen_inputs)
        else:
            result = self.forward(*inputs)

        return result

    def __call__(self, *inputs, **kwargs):
        result = self._call_impl(*inputs, **kwargs)
        return result
