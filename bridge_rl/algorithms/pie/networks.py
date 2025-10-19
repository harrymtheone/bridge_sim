from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, recurrent_wrapper, BaseActor


class Mixer(nn.Module):
    def __init__(
            self,
            prop_size: tuple[int, ...],
            depth_channel: int,
            hidden_size: int,
            activation=nn.ELU(),
    ):
        super().__init__()

        self.prop_enc = make_linear_layers(prop_size, 64, 128, activation_func=activation)

        self.depth_enc = nn.Sequential(
            nn.Conv2d(in_channels=depth_channel, out_channels=32, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size, num_layers=1)

    def forward(self, prop, depth, hidden_states):
        prop_latent = recurrent_wrapper(self.prop_enc, prop)
        depth_latent = recurrent_wrapper(self.depth_enc, depth)
        x = torch.cat((prop_latent, depth_latent), dim=-1)
        return self.gru(x, hidden_states)

    def init_hidden_states(self, num_envs: int, device: torch.device):
        return torch.zeros(self.gru.num_layers, num_envs, self.gru.hidden_size, dtype=torch.float, device=device)


class EstimatorVAE(nn.Module):
    def __init__(
            self,
            input_size: int,
            len_latent_z: int,
            len_latent_hmap: int,
            ot1_size: int,
            scan_size: int,
            activation=nn.ELU(),
    ):
        super().__init__()
        self.len_latent_hmap = len_latent_hmap

        self.encoder = make_linear_layers(input_size, 128, activation_func=activation)

        self.mlp_vel = nn.Linear(input_size, 3)
        self.mlp_vel_logvar = nn.Linear(input_size, 3)
        self.mlp_z = nn.Linear(input_size, len_latent_z + len_latent_hmap)
        self.mlp_z_logvar = nn.Linear(input_size, len_latent_z + len_latent_hmap)

        self.ot1_predictor = make_linear_layers(3 + len_latent_z + len_latent_hmap, 128, ot1_size,
                                                activation_func=activation, output_activation=False)

        self.hmap_recon = make_linear_layers(len_latent_hmap, 256, scan_size,
                                             activation_func=activation, output_activation=False)

    def forward(self, mixer_out, sample=True):
        mu_vel = self.mlp_vel(mixer_out)
        logvar_vel = self.mlp_vel_logvar(mixer_out)
        mu_z = self.mlp_z(mixer_out)
        logvar_z = self.mlp_z_logvar(mixer_out)

        vel = self.reparameterize(mu_vel, logvar_vel) if sample else mu_vel
        z = self.reparameterize(mu_z, logvar_z) if sample else mu_z

        ot1 = self.ot1_predictor(torch.cat([vel, z], dim=-1))
        hmap = self.hmap_recon(z[:, :, -self.len_latent_hmap:])

        return vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class Actor(BaseActor):
    is_recurrent = False

    def __init__(
            self,
            prop_size: int,
            len_latent: int,
            actor_hidden_dims: tuple[int, ...],
            action_size: int,
            activation=nn.ELU(),
    ):
        super().__init__(action_size=action_size)

        self.actor = make_linear_layers(
            prop_size + 3 + len_latent, *actor_hidden_dims, action_size,
            activation_func=activation,
            output_activation=False,
        )

    def forward(self, x) -> torch.Tensor:
        return self.actor(x)

    def sample_actions(self, x):
        self.distribution = Normal(self.forward(x), torch.exp(self.log_std))
        return self.distribution.sample()
