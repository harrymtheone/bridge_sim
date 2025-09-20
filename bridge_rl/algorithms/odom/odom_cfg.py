from __future__ import annotations

from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms import UniversalProprio, UniversalCriticObs, PPOCfg
from . import OdomPPO


@configclass
class OdomCfg(PPOCfg):
    class_type: type = OdomPPO

    # network params
    actor_gru_hidden_size: int = 128
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)

    observations: dict[str, ObservationGroupCfg] = {
        'proprio': UniversalProprio(),

        # 'depth': ObservationGroupCfg(
        #     terms={
        #         'front': ObservationTermCfg(
        #             func=mdp.obs.image,
        #             params=dict(sensor_cfg=MISSING, data_type="depth"),
        #             # noise=GaussianNoiseCfg(mean=0., std=0.05),
        #             # modifiers=  # TODO: we need gaussian filer here
        #         ),
        #         'back': ObservationTermCfg(
        #             func=mdp.obs.image,
        #             params=dict(sensor_cfg=MISSING, data_type="depth"),
        #             # noise=GaussianNoiseCfg(mean=0., std=0.05),
        #             # modifiers=  # TODO: we need gaussian filer here
        #         )
        #     }
        # ),

        # 'scan_edge': ObservationGroupCfg(
        #     terms={
        #         'scan_edge': ObservationTermCfg(
        #             func=mdp.obs.height_scan_with_edge,
        #             params=MISSING,
        #         )
        #     }
        # ),

        # 'est_gt': ObservationGroupCfg(
        #     terms={
        #         'base_lin_vel': ObservationTermCfg(
        #             func=mdp.obs.base_lin_vel,
        #         )
        #     }
        # ),

        'critic_obs': UniversalCriticObs(),
    }
