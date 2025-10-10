from .base_cfg import T1ActionsCfg, T1ArticulationCfg, T1CommandsCfg, T1CurriculumCfg, T1EventCfg, T1MotionGeneratorCfg, T1SceneCfg, T1SceneWithDepthCfg, T1TerminationsCfg

from .dream import dream_tasks
from .odom import odom_tasks
from .pie import t1_pie_tasks

t1_tasks = {}
t1_tasks.update(dream_tasks)
t1_tasks.update(odom_tasks)
t1_tasks.update(t1_pie_tasks)
