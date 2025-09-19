from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

from bridge_env import mdp
from bridge_rl.algorithms import UniversalProprioWithPhase, UniversalCriticObs
from bridge_rl.algorithms.ppo import PPOCfg
from . import OdomPPO


@configclass
class Depth(ObservationGroupCfg):
    d435 = ObservationTermCfg(
        func=mdp.obs.image,
        params=dict(sensor_cfg=MISSING, data_type="depth"),
        noise=GaussianNoiseCfg(mean=0., std=0.05),
        # modifiers=  # TODO: we need gaussian filer here
    )


@configclass
class Scan(ObservationGroupCfg):
    scan = ObservationTermCfg(func=mdp.obs.height_scan_1d, params=MISSING, noise=GaussianNoiseCfg(mean=0., std=0.05))


@configclass
class Priv(ObservationGroupCfg):
    lin_vel = ObservationTermCfg(func=mdp.obs.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))


@configclass
class OdomObservationsCfg:
    proprio: UniversalProprioWithPhase = UniversalProprioWithPhase(enable_corruption=True)

    # depth: Depth = Depth(enable_corruption=True)

    scan: Scan = Scan(enable_corruption=True)

    priv = Priv(enable_corruption=True)

    critic_obs: UniversalCriticObs = UniversalCriticObs(
        enable_corruption=True,
        history_length=50,
        flatten_history_dim=False,
    )


@configclass
class OdomCfg(PPOCfg):
    observation_cfg: OdomObservationsCfg = OdomObservationsCfg()

    class_type: type = OdomPPO

    # network params
    actor_gru_hidden_size: int = 128
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)
