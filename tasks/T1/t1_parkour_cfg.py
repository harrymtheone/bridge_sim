import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.utils import configclass

from bridge_env import mdp


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
        # prim_path="{ENV_REGEX_NS}/Robot/.*foot_link", update_period=0.0, history_length=0, debug_vis=True
        prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=0, debug_vis=True
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
