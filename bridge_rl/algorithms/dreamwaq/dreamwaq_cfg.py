from isaaclab.envs import mdp
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg


@configclass
class DreamwaqObservationsCfg:

    @configclass
    class Proprio(ObservationGroupCfg):
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2))

        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05))
        
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01))

        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5))

        last_action = ObservationTermCfg(func=mdp.last_action)

    proprio: Proprio = Proprio()