from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms import PPOCfg
from . import DreamWaQ
from ..templates import Proprio, UniversalCriticObs


@configclass
class DreamWaQObservationsCfg:
    @configclass
    class Scan(ObservationGroupCfg):
        scan = ObservationTermCfg(func=mdp.obs.height_scan, params=MISSING)

    proprio: Proprio = Proprio(enable_corruption=True)

    critic_obs: UniversalCriticObs = UniversalCriticObs(
        enable_corruption=True,
        history_length=50,
        flatten_history_dim=False,
    )

    scan: Scan = Scan()


@configclass
class DreamWaQCfg(PPOCfg):
    observation_cfg: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    class_type: type = DreamWaQ

    # network parameters
    use_recurrent_policy: bool = True
    num_gru_layers: int = 1
    gru_hidden_size: int = 128

    vae_hidden_size: int = 128
    encoder_output_size: int = 67  # 3 (velocity) + 64 (latent)

    # Actor/Critic network parameters
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)

    # DreamWAQ specific parameters
    symmetry_loss_coef: float = 0.1
    estimation_loss_coef: float = 1.0
    prediction_loss_coef: float = 1.0
    vae_loss_coef: float = 1.0
    update_estimation: bool = True
