from .base_cfg import T1ArticulationCfg, T1SceneCfg, ActionsCfg, TerminationsCfg, EventCfg, CommandsCfg

from .t1_flat_env_cfg import T1FlatEnvCfg
from .t1_parkour_cfg import T1ParkourEnvCfg

from .tasks import *

t1_tasks = {
    't1_dream_flat': T1FlatDreamWaqTaskCfg,
    't1_dream_parkour': T1FlatDreamWaqTaskCfg,

    't1_odom_parkour': T1ParkourOdomPPOTaskCfg,

}

