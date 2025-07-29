from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from bridge_rl.algorithms.ppo import PPOCfg
from . import DreamWaQ


@configclass
class DreamWaQObservationsCfg:
    @configclass
    class Proprio(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))

        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))

        commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))

        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))

        last_action = ObservationTermCfg(func=mdp.last_action)

    proprio: Proprio = Proprio(
        enable_corruption=True
    )

    @configclass
    class CriticObs(ObservationGroupCfg):
        commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)

        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

        last_action = ObservationTermCfg(func=mdp.last_action)

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)

        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)

        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

    critic_obs: CriticObs = CriticObs(
        enable_corruption=True,
        history_length=50,
        flatten_history_dim=False,
    )

    @configclass
    class Scan(ObservationGroupCfg):
        scan = ObservationTermCfg(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg(name="height_scanner"), "offset": 0.7})

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
    actor_hidden_dims: tuple = (256, 128, 64)
    critic_hidden_dims: tuple = (512, 256, 128)

    # DreamWAQ specific parameters
    symmetry_loss_coef: float = 0.1
    estimation_loss_coef: float = 1.0
    prediction_loss_coef: float = 1.0
    vae_loss_coef: float = 1.0
    update_estimation: bool = True
