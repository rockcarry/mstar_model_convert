from calibrator_custom.versions import VERSION

if VERSION[:2] not in ['1.', 'Q_']:
    from calibrator_custom import py_wrapper

def float_simulator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._floatsim import float_simulator
        return float_simulator(*args, **kwargs)
    else:
        return py_wrapper.simulator(*args, **kwargs)

def cmodel_float_simulator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._cmdfloatsim import cmodel_float_simulator
        return cmodel_float_simulator(*args, **kwargs)
    else:
        return py_wrapper.simulator(*args, **kwargs)

def fixed_wipu_simulator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._fixedwipusim import fixed_wipu_simulator
        return fixed_wipu_simulator(*args, **kwargs)
    else:
        return py_wrapper.simulator(*args, **kwargs)

def fixed_simulator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._fixedsim import fixed_simulator
        return fixed_simulator(*args, **kwargs)
    else:
        return py_wrapper.simulator(*args, **kwargs)

def offline_simulator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._offlinesim import offline_simulator
        return offline_simulator(*args, **kwargs)
    else:
        return py_wrapper.simulator(*args, **kwargs)

def calibrator(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._calibrator import calibrator
        return calibrator(*args, **kwargs)
    else:
        return py_wrapper.calibrator(*args, **kwargs)

def converter(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._calibrator import converter
        return converter(*args, **kwargs)
    else:
        return py_wrapper.converter(*args, **kwargs)

def compiler(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from ._compiler import compiler
        return compiler(*args, **kwargs)
    else:
        return py_wrapper.compiler(*args, **kwargs)

def printf(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from calibrator_custom import tool
        return tool.printf(*args, **kwargs)
    else:
        return py_wrapper.printf(*args, **kwargs)

def get_conv_schedule(*args, **kwargs):
    if VERSION[:2] in ['1.', 'Q_']:
        from calibrator_custom import tool
        return tool.get_conv_schedule(*args, **kwargs)
    else:
        return py_wrapper.get_conv_schedule(*args, **kwargs)
