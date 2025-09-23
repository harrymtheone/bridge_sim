from __future__ import annotations

import os

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from bridge_env import mdp, BRIDGE_ROBOTS_DIR
from bridge_rl.algorithms import DreamWaQCfg
from bridge_rl.runners import RLTaskCfg


@configclass
class T1ArticulationCfg(ArticulationCfg):
    prim_path = "{ENV_REGEX_NS}/Robot"

    collision_group = 0
    debug_vis = False

    spawn = UsdFileCfg(
        usd_path=os.path.join(BRIDGE_ROBOTS_DIR, "T1/legs/t1.usd"),
        rigid_props=RigidBodyPropertiesCfg(
            # disable_gravity=True,
            # linear_damping=0.,
            # angular_damping=0.,
            max_linear_velocity=1000.,
            max_angular_velocity=1000.,
        ),
        activate_contact_sensors=True,  # must be enabled if contact sensors are enabled
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    )

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0., 0., 0.7),
        rot=(1., 0., 0., 0.),  # w, x, y, z
        lin_vel=(0., 0., 0.),
        ang_vel=(0., 0., 0.),

        joint_pos={
            'Waist': 0.,
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.,
            'Left_Hip_Yaw': 0.,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': -0.25,
            'Left_Ankle_Roll': 0.,
            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.,
            'Right_Hip_Yaw': 0.,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': -0.25,
            'Right_Ankle_Roll': 0.,
        },
        joint_vel={".*": 0.0},  # the key can be regular expression, here ".*" matches all joint names
    )

    soft_joint_pos_limit_factor = 0.9

    actuators = {}  # initialized in __post_init__

    def __post_init__(self):
        stiffness = {
            'Waist': 100,
            'Hip_Roll': 150, 'Hip_Yaw': 150, 'Hip_Pitch': 150, 'Knee_Pitch': 180, 'Ankle_Roll': 50, 'Ankle_Pitch': 50,
        }

        damping = {
            'Head': 1,
            'Hip_Roll': 8.0, 'Hip_Yaw': 4.0, 'Hip_Pitch': 8, 'Knee_Pitch': 8.0, 'Ankle_Roll': 1.0, 'Ankle_Pitch': 1.0,
            'Shoulder_Pitch': 3, 'Shoulder_Roll': 3, 'Elbow_Pitch': 3, 'Elbow_Yaw': 3, 'Waist': 3.0  # not used yet, set randomly
        }

        for j_name in stiffness:
            self.actuators[j_name] = DelayedPDActuatorCfg(
                joint_names_expr=[f".*{j_name}.*"],
                stiffness=stiffness[j_name],  # Kp
                damping=damping[j_name],  # Kd
                armature=0.01,
                friction=0.,
                min_delay=0,
                max_delay=5,
            )


@configclass
class T1Rewards:
    # -- task
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.rew.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.rew.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewardTermCfg(
        func=mdp.rew.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewardTermCfg(
        func=mdp.rew.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # -- penalties
    flat_orientation_l2 = RewardTermCfg(func=mdp.rew.flat_orientation_l2, weight=0.0)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.rew.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.rew.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewardTermCfg(func=mdp.rew.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.rew.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewardTermCfg(func=mdp.rew.action_rate_l2_v2, weight=-0.01)

    # Penalize ankle joint limits
    dof_pos_limits = RewardTermCfg(
        func=mdp.rew.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )

    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewardTermCfg(
        func=mdp.rew.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Roll", ".*_Hip_Yaw"])},
    )

    termination_penalty = RewardTermCfg(func=mdp.rew.is_terminated, weight=-200.0)


@configclass
class T1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: T1Rewards = T1Rewards()

    motion_generators = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = T1ArticulationCfg()
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*Hip.*", ".*Knee.*"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*Hip.*", ".*Knee.*", ".*Ankle.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"


@configclass
class T1PPORoughTaskCfg(RLTaskCfg):
    """Complete task configuration for T1 Rough PPO."""

    # Environment configuration
    env = T1RoughEnvCfg()

    only_positive_reward = False
    # only_positive_reward_until = None

    # Algorithm configuration  
    algorithm = DreamWaQCfg()
    algorithm.observations.scan.scan.params = dict(sensor_cfg=SceneEntityCfg("height_scanner"), offset=-0.7)
    algorithm.observations.proprio.phase_command = None
    algorithm.observations.critic_obs.phase_command = None

    # Training parameters
    max_iterations = 10000
    num_steps_per_update = 24  # Number of environment steps per policy update

    save_interval = 100
