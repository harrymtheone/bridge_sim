from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bridge_env.envs import mdp
from bridge_rl.algorithms.templates import Proprio, UniversalCriticObs


@configclass
class OdomObservationsCfg:
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
        scan = ObservationTermCfg(func=mdp.obs.height_scan, params=MISSING)

    proprio: Proprio = Proprio(enable_corruption=True)

    depth: Depth = Depth(enable_corruption=True)

    scan: Scan = Scan()

    critic_obs: UniversalCriticObs = UniversalCriticObs(
        enable_corruption=True,
        history_length=50,
        flatten_history_dim=False,
    )
