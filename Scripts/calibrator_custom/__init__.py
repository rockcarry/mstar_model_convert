from calibrator_custom.versions import VERSION as __version__
import os
import platform
modules = os.listdir(__file__[:-12])
modules = [i.split('.')[0] for i in modules]
from calibrator_custom import tool
from calibrator_custom.SIM_simulator import SIM_Simulator
for i, _ in enumerate(modules):
    if platform.architecture()[0][:2] == '64':
        from calibrator_custom.SIM_calibrator import SIM_Calibrator
        if '_floatsim' in modules:
            from ._floatsim import float_simulator
            modules.remove('_floatsim')
        elif '_calibrator' in modules:
            from ._calibrator import calibrator
            modules.remove('_calibrator')
        elif '_converter' in modules:
            from ._converter import converter
            modules.remove('_converter')
    elif platform.architecture()[0][:2] == '32':
        if '_fixedsim' in modules:
            from ._fixedsim import fixed_simulator
            modules.remove('_fixedsim')
        elif '_offlinesim' in modules:
            from ._offlinesim import offline_simulator
            modules.remove('_offlinesim')
