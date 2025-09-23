from .t1_odom_vae_flat_cfg import T1OdomVAEFlatTaskCfg
from .t1_odom_vae_pyramid_cfg import T1OdomVAEPyramidTaskCfg

from .t1_odom_vae_flat_no_phase_cfg import T1OdomVAEFlatNoPhaseTaskCfg

from .t1_rough_ppo_cfg import T1PPORoughTaskCfg

odom_tasks = {
    't1_odom_vae_flat': T1OdomVAEFlatTaskCfg,
    't1_odom_vae_pyramid': T1OdomVAEPyramidTaskCfg,

    't1_odom_vae_flat_no_phase': T1OdomVAEFlatNoPhaseTaskCfg,

    't1_odom_vae_rough_rsl': T1PPORoughTaskCfg,
}

