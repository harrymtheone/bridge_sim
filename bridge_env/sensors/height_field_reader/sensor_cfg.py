from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass

from . import HeightReader


@configclass
class HeightReaderCfg(SensorBaseCfg):
    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type = HeightReader

    alignment: Literal["base", "yaw"] = "yaw"

    pattern_cfg: PatternBaseCfg = MISSING

    offset: OffsetCfg = OffsetCfg()

    interpolation: Literal["average", "minimum"] = "minimum"

    use_guidance: bool = False


# @configclass
# class HeightEdgeScannerCfg(HeightScannerCfg):
#     class_type = HeightEdgeReader
