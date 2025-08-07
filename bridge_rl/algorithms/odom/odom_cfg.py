from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from bridge_env.envs import mdp
from bridge_rl.algorithms.templates import Proprio


@configclass
class OdomObservationsCfg:
    proprio: Proprio = Proprio(
        enable_corruption=True
    )

    @configclass
    class Depth(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.image, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))

    depth: Depth = Depth(
        enable_corruption=True
    )
