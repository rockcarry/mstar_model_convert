import platform
from calibrator_custom.versions import VERSION as __version__
from calibrator_custom.runtime_loader import *
from calibrator_custom import utils
from calibrator_custom.SIM_simulator import SIM_Simulator
if platform.architecture()[0][:2] == '64':
    from calibrator_custom.SIM_calibrator import SIM_Calibrator
    from calibrator_custom.RPC_simulator import RPC_Simulator
    from calibrator_custom import sgs_chalk
