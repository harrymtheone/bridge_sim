from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from bridge_env.managers.motion_generator import MotionTermCfg


@configclass
class BridgeEnvCfg(ManagerBasedRLEnvCfg):
    motion_generators: dict[str, MotionTermCfg] = {}
