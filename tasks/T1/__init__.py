from .base_cfg import T1ArticulationCfg, T1SceneCfg, T1ActionsCfg, T1TerminationsCfg, T1EventCfg, T1MotionGeneratorCfg, T1CommandsCfg

from .dream import dream_tasks
from .odom import odom_tasks

t1_tasks = {}
t1_tasks.update(dream_tasks)
t1_tasks.update(odom_tasks)
