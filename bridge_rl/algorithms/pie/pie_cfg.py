from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms import UniversalProprio, UniversalCriticObs, PPOCfg
from . import PIE


@configclass
class OdomObservationCfg:
    @configclass
    class Depth(ObservationGroupCfg):
        depth_front = ObservationTermCfg(
            func=mdp.obs.image,
            params=dict(sensor_cfg=MISSING, data_type="depth"),
            # noise=GaussianNoiseCfg(mean=0., std=0.05),
            # modifiers=  # TODO: we need gaussian filer here
        )

    @configclass
    class EstGT(ObservationGroupCfg):
        vel = ObservationTermCfg(func=mdp.obs.base_lin_vel)

    proprio = UniversalProprio(
        enable_corruption=True,
    )

    depth = Depth()

    est_gt = EstGT()

    critic_obs: UniversalCriticObs = UniversalCriticObs(
        enable_corruption=True,
    )


@configclass
class PIECfg(PPOCfg):
    class_type: type = PIE

    observations = OdomObservationCfg()

    # -- Actor Critic
    actor_gru_hidden_size: int = 128
    actor_gru_num_layers: int = 2
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)

    # -- VAE
    vae_latent_size: int = 16
    vae_loss_z_coef: float = 1.0
    vae_loss_vel_coef: float = 1.0
