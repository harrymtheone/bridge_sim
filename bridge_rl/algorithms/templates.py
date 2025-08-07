from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from bridge_env.envs import mdp


@configclass
class Proprio(ObservationGroupCfg):
    base_ang_vel = ObservationTermCfg(func=mdp.obs.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))

    projected_gravity = ObservationTermCfg(func=mdp.obs.projected_gravity, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))

    commands = ObservationTermCfg(func=mdp.obs.generated_commands, params={"command_name": "base_velocity"})

    joint_pos = ObservationTermCfg(func=mdp.obs.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))

    joint_vel = ObservationTermCfg(func=mdp.obs.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))

    last_action = ObservationTermCfg(func=mdp.obs.last_action)


@configclass
class UniversalCriticObs(ObservationGroupCfg):
    base_lin_vel = ObservationTermCfg(func=mdp.obs.base_lin_vel)

    base_ang_vel = ObservationTermCfg(func=mdp.obs.base_ang_vel)

    projected_gravity = ObservationTermCfg(func=mdp.obs.projected_gravity)

    commands = ObservationTermCfg(func=mdp.obs.generated_commands, params={"command_name": "base_velocity"})

    joint_pos = ObservationTermCfg(func=mdp.obs.joint_pos_rel)

    joint_vel = ObservationTermCfg(func=mdp.obs.joint_vel_rel)

    last_action = ObservationTermCfg(func=mdp.obs.last_action)

    foot_scan_l = ObservationTermCfg(func=mdp.obs.foothold_1d, params=MISSING)
    foot_scan_r = ObservationTermCfg(func=mdp.obs.foothold_1d, params=MISSING)

    feet_is_contact = ObservationTermCfg(func=mdp.obs.link_is_contact, params=MISSING)
