from .base_cfg import T1ArticulationCfg, T1SceneCfg, T1ActionsCfg, T1TerminationsCfg, T1EventCfg, T1MotionGeneratorCfg, T1CommandsCfg

from .t1_flat_env_cfg import T1FlatEnvCfg
from .t1_parkour_cfg import T1ParkourEnvCfg
from .t1_full_cfg import T1FullEnvCfg

from .tasks import *

t1_tasks = {
    't1_dream_flat': T1FlatDreamWaqTaskCfg,
    't1_dream_parkour': T1FlatDreamWaqTaskCfg,

    't1_odom_parkour': T1ParkourOdomPPOTaskCfg,

    't1_full': T1FullEnvCfg,
}

