from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg, ObservationGroupCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms import UniversalProprioWithPhase, UniversalCriticObsWithPhase, PPOCfg
from . import PIE


@configclass
class OdomObservationCfg:
    @configclass
    class Scan(ObservationGroupCfg):
        scan = ObservationTermCfg(func=mdp.obs.height_scan_1d, params=MISSING)

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

    proprio = UniversalProprioWithPhase(enable_corruption=True)

    prop_next = UniversalProprioWithPhase(enable_corruption=False)  # see task_runner and process_env_step
    prop_next.phase_command = None
    prop_next.vel_command = None
    prop_next.last_action = None

    depth = Depth()

    scan = Scan()

    est_gt = EstGT()

    critic_obs = UniversalCriticObsWithPhase(enable_corruption=True)


@configclass
class PIECfg(PPOCfg):
    class_type: type = PIE

    observations = OdomObservationCfg()

    # -- Encoder
    mixer_depth_channel: int = 1
    mixer_hidden_size: int = 256

    # -- VAE
    len_latent_z: int = 16
    len_latent_hmap: int = 16
    len_command: int = 5
    vae_latent_size: int = 16
    vae_loss_z_coef: float = 1.0
    vae_loss_vel_coef: float = 1.0

    # -- Actor Critic
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)
