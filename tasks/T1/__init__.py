from .base_cfg import T1ArticulationCfg, T1SceneCfg, T1ActionsCfg, T1TerminationsCfg, T1EventCfg, T1MotionGeneratorCfg, T1CommandsCfg

from .t1_flat_dream_cfg import T1FlatEnvCfg, T1DreamWaqFlatTaskCfg
from .t1_parkour_cfg import T1ParkourEnvCfg
from .t1_full_cfg import T1FullEnvCfg

t1_tasks = {
    't1_dream_flat': T1DreamWaqFlatTaskCfg,
    't1_dream_parkour': T1DreamWaqFlatTaskCfg,

    # 't1_odom_parkour': T1ParkourOdomPPOTaskCfg,

    # 't1_full': T1FullEnvCfg,
}

