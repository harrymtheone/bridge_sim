from __future__ import annotations

import functools
import operator
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from bridge_rl.algorithms.ppo import BaseActor, BaseRecurrentActor, BaseCritic
from bridge_rl.algorithms.utils import make_linear_layers, recurrent_wrapper


class VAE(nn.Module):
    """Variational Autoencoder for DreamWAQ privileged information estimation."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        activation = nn.ELU()

        self.mlp_mu = make_linear_layers(input_size, hidden_size, output_size,
                                         activation_func=activation,
                                         output_activation=False)

        self.mlp_logvar = make_linear_layers(input_size, hidden_size, output_size,
                                             activation_func=activation,
                                             output_activation=False)

        self.decoder = make_linear_layers(output_size, 64, input_size,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, obs_enc, mu_only: bool = False):
        """Forward pass through VAE.

        Args:
            obs_enc: Encoded observations
            mu_only: If True, only return mean (for inference)

        Returns:
            If mu_only: mean only
            Otherwise: (reconstructed_obs, mean, log_variance)
        """
        if mu_only:
            return self.mlp_mu(obs_enc)

        est_mu = self.mlp_mu(obs_enc)
        est_logvar = self.mlp_logvar(obs_enc)
        ot1 = self.decoder(self.reparameterize(est_mu, est_logvar))
        return ot1, est_mu, est_logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class DreamWaQActor(BaseActor):
    is_recurrent = False

    def __init__(self,
                 obs_size: int,
                 action_size: int,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64),
                 encoder_output_size: int = 67,
                 channel_size: int = 16):
        super().__init__(action_size)

        self.activation = nn.ELU()
        self.encoder_output_size = encoder_output_size

        # Observation encoder (1D conv for proprioceptive history)
        self.obs_enc = nn.Sequential(
            nn.Conv1d(in_channels=obs_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
            self.activation,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
            self.activation,
            nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
            self.activation,
            nn.Flatten()
        )

        # VAE for privileged information estimation
        self.vae = VAE(input_size=8 * channel_size, output_size=encoder_output_size)

        # Actor network
        self.actor_backbone = make_linear_layers(
            obs_size + encoder_output_size,
            *hidden_dims,
            action_size,
            activation_func=self.activation,
            output_activation=False
        )

        # Disable args validation for speedup
        Normal.set_default_validate_args = False

    def act(self, obs, eval_: bool = False, **kwargs) -> torch.Tensor:
        """Generate actions from observations."""
        # Encode proprioceptive history
        obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))

        # Get VAE estimation
        est_mu = self.vae(obs_enc, mu_only=True)

        # Concatenate current proprioception with VAE output
        actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, **kwargs) -> None:
        """Generate actions during training."""
        # Encode proprioceptive history
        obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))

        # Get VAE estimation
        est_mu = self.vae(obs_enc, mu_only=True)

        # Concatenate current proprioception with VAE output
        actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
        mean = self.actor_backbone(actor_input)

        self.distribution = Normal(mean, torch.exp(self.log_std))

    def estimate(self, obs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate privileged information using VAE.

        Returns:
            (reconstructed_obs, estimated_velocity, mean, log_variance)
        """
        obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
        ot1, est_mu, est_logvar = self.vae(obs_enc, mu_only=False)
        return ot1, est_mu[..., :3], est_mu, est_logvar  # First 3 elements are velocity


class DreamWaQRecurrentActor(BaseRecurrentActor):
    is_recurrent = True

    def __init__(self,
                 prop_shape: tuple[int, ...],
                 num_gru_layers: int,
                 gru_hidden_size: int,
                 actor_hidden_dims: Tuple[int, ...],
                 encoder_output_size: int,
                 action_size: int):
        super().__init__(action_size)

        self.activation = nn.ELU()
        self.encoder_output_size = encoder_output_size

        if len(prop_shape) == 1:
            # without history
            prop_size = prop_shape[0]
        elif len(prop_shape) == 2:
            # with history
            prop_size = prop_shape[0] * prop_shape[1]
        else:
            raise ValueError(f'prop_shape {prop_shape} is not valid!')

        self.gru = nn.GRU(input_size=prop_size, hidden_size=gru_hidden_size, num_layers=num_gru_layers)
        self.hidden_states = None

        self.vae = VAE(input_size=gru_hidden_size, output_size=encoder_output_size)

        self.actor_backbone = make_linear_layers(
            encoder_output_size + prop_size,
            *actor_hidden_dims,
            action_size,
            activation_func=self.activation,
            output_activation=False
        )

    def act(self,
            obs: dict[str, torch.Tensor],
            use_estimated_value=True,
            eval_: bool = False,
            **kwargs) -> torch.Tensor:
        proprio = obs['proprio']

        # Process through GRU
        obs_enc, self.hidden_states = self.gru(proprio.unsqueeze(0), self.hidden_states)

        # Get VAE estimation
        est_mu = self.vae(obs_enc.squeeze(0), mu_only=True)

        # Concatenate proprioception with VAE output
        actor_input = torch.cat([proprio, self.activation(est_mu)], dim=1)
        mean = self.actor_backbone(actor_input)

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self,
                  obs: dict[str, torch.Tensor],
                  hidden_states: torch.Tensor = None,
                  **kwargs) -> None:

        proprio = obs['proprio']

        obs_enc, _ = self.gru(proprio, hidden_states)

        # Get VAE estimation
        est_mu = recurrent_wrapper(self.vae.forward, obs_enc, mu_only=True)

        # Concatenate proprioception with VAE output
        actor_input = torch.cat([proprio, self.activation(est_mu)], dim=2)
        mean = recurrent_wrapper(self.actor_backbone.forward, actor_input)

        self.distribution = Normal(mean, torch.exp(self.log_std))

    def estimate(self, obs, hidden_states: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate privileged information using VAE.

        Returns:
            (reconstructed_obs, estimated_velocity, mean, log_variance)
        """
        obs_enc, _ = self.gru(obs.proprio, hidden_states)
        ot1, est_mu, est_logvar = recurrent_wrapper(self.vae.forward, obs_enc, mu_only=False)
        return ot1, est_mu[..., :3], est_mu, est_logvar


class DreamWaQCritic(BaseCritic):
    def __init__(self,
                 critic_obs_shape: tuple[int, int],
                 scan_shape: tuple[int, ...],
                 hidden_dims: Tuple[int, ...] = (512, 256, 128)):
        super().__init__()

        activation = nn.ELU()

        if len(critic_obs_shape) in (1, 2):
            critic_obs_size = critic_obs_shape[-1]
        else:
            raise ValueError(f'critic_obs_shape {critic_obs_shape} is not valid!')

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
        scan_size = functools.reduce(operator.mul, scan_shape, 1)
        self.scan_enc = make_linear_layers(scan_size, 256, 64, activation_func=activation)

        # Value function network
        self.critic = make_linear_layers(
            128 + 64,
            *hidden_dims,
            1,
            activation_func=activation,
            output_activation=False
        )

    def evaluate(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:

        if obs['critic_obs'].ndim == 3:
            # Non-recurrent case
            priv_latent = self.priv_enc(obs['critic_obs'].transpose(1, 2))
            scan_enc = self.scan_enc(obs['scan'].flatten(1))
            return self.critic(torch.cat([priv_latent, scan_enc], dim=1))
        else:
            return recurrent_wrapper(self.evaluate, obs)
