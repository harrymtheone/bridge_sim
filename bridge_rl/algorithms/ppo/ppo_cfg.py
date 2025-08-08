from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils import configclass

from . import PPO


@configclass
class PPOCfg:
    observation_cfg: Any = MISSING

    class_type: type = PPO

    # RL parameters
    gamma: float = 0.99
    lam: float = 0.95

    # PPO parameters
    clip_param: float = 0.2
    num_mini_batches: int = 4
    num_learning_epochs: int = 5
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    # Noise parameters
    entropy_coef: float = 0.01
    init_noise_std: float = 1.0
    noise_std_range: tuple[float, float] = (0.3, 1.0)
    continue_from_last_std: bool = False

    # Learning rate scheduling
    learning_rate: float = 1e-3
    learning_rate_schedule: Literal["adaptive", "fixed"] = "adaptive"
    desired_kl: float | None = 0.01

    # Mixed precision training
    use_amp: bool = False

    # Gradient clipping
    max_grad_norm: float | None = 1.0

    num_steps_per_update: int = MISSING
    """ Note: this values automatically filled by runner cfg """
