from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from bridge_env.sensors.ray_caster import FootholdRayCaster


@configclass
class FootholdRayCasterCfg(RayCasterCfg):
    class_type = FootholdRayCaster

    reading_bias_z: float = MISSING

    ray_alignment: str = "foothold"
