import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.utils import configclass

from bridge_env import mdp
from bridge_rl.algorithms.dreamwaq import DreamWaQCfg


@configclass
class T1ArticulationCfg(ArticulationCfg):
    prim_path = "{ENV_REGEX_NS}/Robot"

    collision_group = 0
    debug_vis = False

    spawn = UsdFileCfg(
        usd_path='/home/harry/projects/bridge_sim_v2/robots/T1/usd/usd.usd',
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.,
            angular_damping=0.,
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
            'AAHead_yaw': 0.,
            'Head_pitch': 0.,

            'Left_Shoulder_Pitch': 0.,
            'Left_Shoulder_Roll': -1.3,
            'Left_Elbow_Pitch': 0.,
            'Left_Elbow_Yaw': -1.,
            'Right_Shoulder_Pitch': 0.,
            'Right_Shoulder_Roll': 1.3,
            'Right_Elbow_Pitch': 0.,
            'Right_Elbow_Yaw': 1.,
            'Waist': 0.,

            'Left_Hip_Pitch': -0.2 - 0.1974,
            'Left_Hip_Roll': 0.,
            'Left_Hip_Yaw': 0.,
            'Left_Knee_Pitch': 0.4 + 0.3948,
            'Left_Ankle_Pitch': -0.25 - 0.1974,
            'Left_Ankle_Roll': 0.,
            'Right_Hip_Pitch': -0.2 - 0.1974,
            'Right_Hip_Roll': 0.,
            'Right_Hip_Yaw': 0.,
            'Right_Knee_Pitch': 0.4 + 0.3948,
            'Right_Ankle_Pitch': -0.25 - 0.1974,
            'Right_Ankle_Roll': 0.,
        },
        joint_vel={".*": 0.0},  # the key can be regular expression, here ".*" matches all joint names
    )

    soft_joint_pos_limit_factor = 1.0  # TODO: not sure if in target joint position mode, the input action got clipped by this value

    actuators = {}  # initialized in __post_init__

    def __post_init__(self):
        stiffness = {
            'Head': 30,
            'Hip_Roll': 150, 'Hip_Yaw': 150, 'Hip_Pitch': 150, 'Knee_Pitch': 180, 'Ankle_Roll': 50, 'Ankle_Pitch': 50,
            'Shoulder_Pitch': 300, 'Shoulder_Roll': 200, 'Elbow_Pitch': 200, 'Elbow_Yaw': 100, 'Waist': 100  # not used yet, set randomly
        }

        damping = {
            'Head': 1,
            'Hip_Roll': 8.0, 'Hip_Yaw': 4.0, 'Hip_Pitch': 8, 'Knee_Pitch': 8.0, 'Ankle_Roll': 1.0, 'Ankle_Pitch': 1.0,
            'Shoulder_Pitch': 3, 'Shoulder_Roll': 3, 'Elbow_Pitch': 3, 'Elbow_Yaw': 3, 'Waist': 3.0  # not used yet, set randomly
        }

        # stiffness = {
        #     'Head': 30,
        #     'Hip_Roll': 55, 'Hip_Yaw': 30, 'Hip_Pitch': 55, 'Knee_Pitch': 100, 'Ankle_Roll': 30, 'Ankle_Pitch': 30,
        #     'Shoulder_Pitch': 300, 'Shoulder_Roll': 200, 'Elbow_Pitch': 200, 'Elbow_Yaw': 100, 'Waist': 200  # not used yet, set randomly
        # }
        #
        # damping = {
        #     'Head': 1,
        #     'Hip_Roll': 3.0, 'Hip_Yaw': 4.0, 'Hip_Pitch': 3, 'Knee_Pitch': 6.0, 'Ankle_Roll': 0.3, 'Ankle_Pitch': 0.3,
        #     'Shoulder_Pitch': 3, 'Shoulder_Roll': 3, 'Elbow_Pitch': 3, 'Elbow_Yaw': 3, 'Waist': 10.0  # not used yet, set randomly
        # }

        for j_name in stiffness:
            # You should understand the difference between implicit and explicit actuator
            self.actuators[j_name] = ImplicitActuatorCfg(
                joint_names_expr=[f".*{j_name}.*"],  # matches joints with name contains j_name
                # effort_limit=?,  # load from USD file
                # velocity_limit=?,  # load from USD file
                stiffness=stiffness[j_name],  # Kp
                damping=damping[j_name],  # Kd
                armature=0.1,
                friction=0.,
                # min_delay=0,
                # max_delay=5,
            )


@configclass
class T1SceneCfg(InteractiveSceneCfg):
    """Configuration for a T1 scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: T1ArticulationCfg = T1ArticulationCfg()

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=0, debug_vis=False
    )

    height_scanner = RayCasterCfg(
        mesh_prim_paths=["/World/defaultGroundPlane"],
        attach_yaw_only=True,
        pattern_cfg=GridPatternCfg(
            resolution=0.05,
            size=(1.8, 3.2),
        ),
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        update_period=0.0,
        history_length=0,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class RewardsCfg:
    # ######### task #########
    track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "tracking_sigma": 5})

    track_ang_vel_z_exp = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "tracking_sigma": 5})

    # ######### penalties #########
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot.*"), "threshold": 1.0},
    )

    # ######### optional penalties #########
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )


@configclass
class EventCfg:
    # startup
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.PhaseCommandCfg(
        base_command_cfg=mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        ),
        num_phase_clock=2,
        period_s=0.7,
        phase_bias=[0, math.pi / 2],
        randomize_start_phase=True,
        stand_walk_switch=True,
    )


@configclass
class T1ParkourDreamwaqCfg(ManagerBasedRLEnvCfg):
    episode_length_s = 40.0

    decimation = 4

    scene = T1SceneCfg(num_envs=16, env_spacing=2.0)

    actions = ActionsCfg()

    rewards = RewardsCfg()

    terminations = TerminationsCfg()

    events = EventCfg()

    commands = CommandsCfg()
