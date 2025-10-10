from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, BaseRecurrentActor


class VAE(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, ...],
            proprio_size: int,
            vae_latent_size: int = 16,
            activation=nn.ELU(),
    ):
        super().__init__()

        input_size = math.prod(input_shape)
        self.encoder = make_linear_layers(input_size, 128, 64,
                                          activation_func=activation)

        self.mlp_vel = nn.Linear(64, 3)
        self.mlp_vel_logvar = nn.Linear(64, 3)

        self.mlp_z = nn.Linear(64, vae_latent_size)
        self.mlp_z_logvar = nn.Linear(64, vae_latent_size)

        self.decoder = make_linear_layers(vae_latent_size + 3, 64, proprio_size,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, x, sample=True):
        enc = self.encoder(x)

        mu_vel, mu_z = self.mlp_vel(enc), self.mlp_z(enc)

        logvar_vel, logvar_z = self.mlp_vel_logvar(enc), self.mlp_z_logvar(enc)

        vel = self.reparameterize(mu_vel, logvar_vel) if sample else mu_vel
        z = self.reparameterize(mu_z, logvar_z) if sample else mu_z

        ot1 = self.decoder(torch.cat([vel, z], dim=-1))

        return vel, z, ot1, mu_vel, logvar_vel, mu_z, logvar_z

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


# class DreamWaQActor(BaseActor):
#     is_recurrent = False
#
#     def __init__(self,
#                  obs_size: int,
#                  action_size: int,
#                  hidden_dims: Tuple[int, ...] = (256, 128, 64),
#                  encoder_output_size: int = 67,
#                  channel_size: int = 16):
#         super().__init__(action_size)
#
#         self.activation = nn.ELU()
#         self.encoder_output_size = encoder_output_size
#
#         # Observation encoder (1D conv for proprioceptive history)
#         self.obs_enc = nn.Sequential(
#             nn.Conv1d(in_channels=obs_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
#             self.activation,
#             nn.Conv1d(in_channels=2 * channel_size, out_channels=4 * channel_size, kernel_size=6, stride=1),
#             self.activation,
#             nn.Conv1d(in_channels=4 * channel_size, out_channels=8 * channel_size, kernel_size=6, stride=1),
#             self.activation,
#             nn.Flatten()
#         )
#
#         # VAE for privileged information estimation
#         self.vae = VAE(input_size=8 * channel_size, output_size=encoder_output_size)
#
#         # Actor network
#         self.actor_backbone = make_linear_layers(
#             obs_size + encoder_output_size,
#             *hidden_dims,
#             action_size,
#             activation_func=self.activation,
#             output_activation=False
#         )
#
#         # Disable args validation for speedup
#         Normal.set_default_validate_args = False
#
#     def act(self, obs, eval_: bool = False, **kwargs) -> torch.Tensor:
#         """Generate actions from observations."""
#         # Encode proprioceptive history
#         obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
#
#         # Get VAE estimation
#         est_mu = self.vae(obs_enc, mu_only=True)
#
#         # Concatenate current proprioception with VAE output
#         actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
#         mean = self.actor_backbone(actor_input)
#
#         if eval_:
#             return mean
#
#         self.distribution = Normal(mean, torch.exp(self.log_std))
#         return self.distribution.sample()
#
#     def train_act(self, obs, **kwargs) -> None:
#         """Generate actions during training."""
#         # Encode proprioceptive history
#         obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
#
#         # Get VAE estimation
#         est_mu = self.vae(obs_enc, mu_only=True)
#
#         # Concatenate current proprioception with VAE output
#         actor_input = torch.cat((obs.proprio, self.activation(est_mu)), dim=1)
#         mean = self.actor_backbone(actor_input)
#
#         self.distribution = Normal(mean, torch.exp(self.log_std))
#
#     def estimate(self, obs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Estimate privileged information using VAE.
#
#         Returns:
#             (reconstructed_obs, estimated_velocity, mean, log_variance)
#         """
#         obs_enc = self.obs_enc(obs.prop_his.transpose(1, 2))
#         ot1, est_mu, est_logvar = self.vae(obs_enc, mu_only=False)
#         return ot1, est_mu[..., :3], est_mu, est_logvar  # First 3 elements are velocity


class DreamWaQRecurrentActor(BaseRecurrentActor):
    is_recurrent = True

    def __init__(
            self,
            prop_shape: tuple[int, ...],
            vae_latent_size: int,
            num_gru_layers: int,
            gru_hidden_size: int,
            actor_hidden_dims: Tuple[int, ...],
            action_size: int
    ):
        super().__init__(action_size)

        self.num_gru_layers = num_gru_layers
        self.gru_hidden_size = gru_hidden_size
        self.activation = nn.ELU()

        prop_size = math.prod(prop_shape)

        self.gru = nn.GRU(input_size=prop_size, hidden_size=gru_hidden_size, num_layers=num_gru_layers)
        self.hidden_states = None

        self.vae = VAE(input_shape=(gru_hidden_size,), proprio_size=prop_size, vae_latent_size=vae_latent_size)

        self.actor_backbone = make_linear_layers(
            vae_latent_size + 3 + prop_size,
            *actor_hidden_dims,
            action_size,
            activation_func=self.activation,
            output_activation=False
        )

    def init_hidden_states(self, num_envs: int, device: torch.device) -> None:
        self.hidden_states = torch.zeros(self.num_gru_layers, num_envs, self.gru_hidden_size, device=device)

    def get_hidden_states(self) -> torch.Tensor:
        return self.hidden_states

    def reset(self, dones: torch.Tensor) -> None:
        self.hidden_states[:, dones] = 0.

    def act(self,
            obs: dict[str, torch.Tensor],
            eval_: bool = False,
            **kwargs):
        proprio = obs['proprio'].unsqueeze(0)

        # Process through GRU
        obs_enc, self.hidden_states = self.gru(proprio, self.hidden_states)

        # Get VAE estimation
        vel, z = self.vae(obs_enc, sample=not eval_)[:2]

        # Concatenate proprioception with VAE output
        mean = self.actor_backbone(torch.cat([vel, z, proprio], dim=-1))

        mean, vel, z = mean.squeeze(0), vel.squeeze(0), z.squeeze(0)

        if eval_:
            return mean, vel, z

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample(), vel, z

    def actor_forward(self, proprio: torch.Tensor, vel: torch.Tensor, z: torch.Tensor):
        mean = self.actor_backbone(torch.cat([vel, z, proprio], dim=-1))
        self.distribution = Normal(mean, torch.exp(self.log_std))

    def vae_forward(
            self,
            obs: dict[str, torch.Tensor],
            hidden_states: torch.Tensor
    ):
        proprio = obs['proprio']
        obs_enc, _ = self.gru(proprio, hidden_states)
        return self.vae(obs_enc)
