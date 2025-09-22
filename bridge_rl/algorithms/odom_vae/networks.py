from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, recurrent_wrapper, BaseRecurrentActor


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

        self.mlp_latent = nn.Linear(64, vae_latent_size)
        self.mlp_latent_logvar = nn.Linear(64, vae_latent_size)

        self.mlp_vel = nn.Linear(64, 3)
        self.mlp_vel_logvar = nn.Linear(64, 3)

        self.decoder = make_linear_layers(vae_latent_size + 3, 64, proprio_size,
                                          activation_func=activation,
                                          output_activation=False)

    def forward(self, x, mu_only=False):
        enc = self.encoder(x)

        mu_z, mu_vel = self.mlp_latent(enc), self.mlp_vel(enc)

        if mu_only:
            return mu_z, mu_vel

        logvar_z, logvar_vel = self.mlp_latent_logvar(enc), self.mlp_vel_logvar(enc)

        z, vel = self.reparameterize(mu_z, logvar_z), self.reparameterize(mu_vel, logvar_vel)

        ot1 = self.decoder(torch.cat([z, vel], dim=1))

        return z, vel, ot1, mu_z, logvar_z, mu_vel, logvar_vel

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


class OdomVAEActor(BaseRecurrentActor):
    is_recurrent = True

    def __init__(
            self,
            prop_shape: tuple[int, ...],
            scan_shape: tuple[int, ...],
            vae_latent_size: int,
            actor_gru_hidden_size: int,
            actor_hidden_dims: tuple[int, ...],
            actor_gru_num_layers: int,
            action_size: int,
            activation=nn.ELU(),
    ):
        super().__init__(action_size=action_size)

        assert len(prop_shape) == 1 and len(scan_shape) == 1
        prop_size, scan_size = prop_shape[0], scan_shape[0]
        self.actor_gru_num_layers = actor_gru_num_layers

        # Encodes height scan or alternative reconstructed scan
        self.scan_encoder = make_linear_layers(
            scan_size, 256, 128,
            activation_func=activation
        )

        # Belief encoder (GRU)
        self.gru = nn.GRU(prop_size + 128, actor_gru_hidden_size, num_layers=actor_gru_num_layers)

        self.vae = VAE(
            input_shape=(actor_gru_hidden_size,),
            proprio_size=prop_size,
            vae_latent_size=vae_latent_size,
        )

        # Actor MLP head
        self.actor = make_linear_layers(
            vae_latent_size + 3 + actor_gru_hidden_size,
            *actor_hidden_dims,
            action_size,
            activation_func=activation,
            output_activation=False,
        )

    def act(self, obs, eval_: bool = False, **kwargs):
        proprio = obs['proprio']
        scan = obs['scan']

        scan_enc = self.scan_encoder(scan)
        x = torch.cat([proprio, scan_enc], dim=1)

        # GRU forward
        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        z, vel = self.vae(x)[:2]

        mean = self.actor(torch.cat([z, vel, x], dim=1))

        if eval_:
            return mean

        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, hidden_states=None, **kwargs):
        proprio = obs['proprio']
        scan = obs['scan']

        scan_enc = recurrent_wrapper(self.scan_encoder.forward, scan.flatten(2))

        x = torch.cat([proprio, scan_enc], dim=2)
        x, _ = self.gru(x, hidden_states)

        z, vel = recurrent_wrapper(self.vae, x)[:2]

        mean = recurrent_wrapper(self.actor.forward, torch.cat([z, vel, x], dim=2))
        self.distribution = Normal(mean, torch.exp(self.log_std))
        return x

    def init_hidden_states(self, num_envs, device):
        self.hidden_states = torch.zeros(self.actor_gru_num_layers, num_envs, self.gru.hidden_size, device=device)
