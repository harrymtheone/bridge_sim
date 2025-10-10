from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms import UniversalProprioWithPhase, UniversalCriticObsWithPhase, PPOCfg
from . import DreamWaQ


@configclass
class DreamWaQObservationsCfg:
    @configclass
    class Scan(ObservationGroupCfg):
        scan = ObservationTermCfg(func=mdp.obs.height_scan_1d, params=MISSING)

    @configclass
    class EstGT(ObservationGroupCfg):
        vel = ObservationTermCfg(func=mdp.obs.base_lin_vel)

    proprio: UniversalProprioWithPhase = UniversalProprioWithPhase(enable_corruption=True)

    critic_obs: UniversalCriticObsWithPhase = UniversalCriticObsWithPhase(
        enable_corruption=True,
        history_length=50,
        flatten_history_dim=False,
    )

    scan: Scan = Scan()

    est_gt: EstGT = EstGT()


@configclass
class DreamWaQCfg(PPOCfg):
    class_type: type = DreamWaQ

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    # -- Actor Critic
    use_recurrent_policy: bool = True
    num_gru_layers: int = 1
    gru_hidden_size: int = 128
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)

    # -- VAE
    vae_latent_size: int = 16
    vel_est_loss_coef: float = 1.0
    ot1_pred_loss_coef: float = 1.0
    kl_coef_vel: float = 1.0
    kl_coef_z: float = 1.0

    # other parameters
    symmetry_loss_coef: float = 1.0
