from __future__ import annotations

from typing import Literal

from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from bridge_env.sensors.ray_caster import FootholdRayCaster


@configclass
class FootholdRayCasterCfg(RayCasterCfg):
    class_type = FootholdRayCaster

    ray_alignment: Literal["base", "yaw", "world", "foothold"] = "base"
