from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from bridge_rl.algorithms import make_linear_layers, recurrent_wrapper


class BaseActor(nn.Module):
    is_recurrent: bool

    def __init__(self, action_size: int):
        super().__init__()

        # Action noise parameter
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.distribution: Optional[Distribution] = None

        # Disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_: bool = False, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Generate actions from observations.

        Args:
            obs: Observations (can be tensor or structured observation)
            eval_: Whether in evaluation mode (if True, return deterministic actions)
            **kwargs: Additional arguments for custom actors

        Returns:
            Actions tensor
        """
        raise NotImplementedError("Subclasses must implement act method")

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of given actions.

        Args:
            actions: Actions tensor

        Returns:
            Log probabilities tensor
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not set. Call act() first.")
        return self.distribution.log_prob(actions).sum(dim=-1, keepdim=True)

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not set. Call act() first.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get standard deviation of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not set. Call act() first.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of current action distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not set. Call act() first.")
        return self.distribution.entropy().sum(dim=-1, keepdim=True)

    def reset_std(self, std: float, device: torch.device) -> None:
        """Reset action standard deviation."""
        new_log_std = torch.log(std * torch.ones_like(self.log_std.data, device=device))
        self.log_std.data = new_log_std.data

    def clip_std(self, min_std: float, max_std: float) -> None:
        self.log_std.data = torch.clamp(self.log_std.data, math.log(min_std), math.log(max_std))


class BaseRecurrentActor(BaseActor):
    """Base template class for recurrent actors.

    Extends BaseActor with recurrent-specific functionality.
    """

    is_recurrent: bool = True

    def init_hidden_states(self, num_envs: int, device: torch.device) -> None:
        raise NotImplementedError

    def get_hidden_states(self) -> torch.Tensor:
        """Get current hidden states.

        Returns:
            Hidden states tensor or None if not initialized
        """
        raise NotImplementedError

    def reset(self, dones: torch.Tensor) -> None:
        """Reset hidden states for done environments.

        Args:
            dones: Boolean tensor indicating which environments are done
        """
        raise NotImplementedError


class PPOCritic(nn.Module):
    def __init__(
            self,
            critic_obs_shape: tuple[int, int],
            scan_shape: tuple[int, ...],
            critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()

        activation = nn.ELU()

        if len(critic_obs_shape) == 2:
            critic_obs_size = critic_obs_shape[-1]
        else:
            raise ValueError(f'critic_obs shape {critic_obs_shape} is not valid!')

        # Proprioceptive history encoder
        self.priv_enc = nn.Sequential(
            nn.Conv1d(in_channels=critic_obs_size, out_channels=64, kernel_size=6, stride=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2),
            activation,
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            activation,
            nn.Flatten()
        )

        # Scan encoder
        scan_size = math.prod(scan_shape)
        self.scan_enc = make_linear_layers(scan_size, 256, 64, activation_func=activation)

        # Value function network
        self.critic = make_linear_layers(
            128 + 64, *critic_hidden_dims, 1,
            activation_func=activation,
            output_activation=False
        )

    def forward(self, critic_obs, scan, **kwargs) -> torch.Tensor:
        priv_latent = recurrent_wrapper(self.priv_enc, critic_obs.transpose(2, 3))
        scan_enc = self.scan_enc(scan.flatten(2))
        return self.critic(torch.cat([priv_latent, scan_enc], dim=-1))
