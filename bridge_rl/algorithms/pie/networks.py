from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, recurrent_wrapper, BaseRecurrentActor


class EstimatorGRU(nn.Module):
    def __init__(
            self,
            prop_shape: tuple[int, ...],
            depth_channel: int,
            hidden_size: int,
            activation=nn.ELU(),
    ):
        super().__init__()

        self.prop_his_enc = nn.Sequential(
            nn.Conv1d(in_channels=prop_size, out_channels=32, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4),
            activation,
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4),
            activation,
            nn.Flatten()
        )

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
        self.hidden_states = None

    def inference(self, prop_his, depth_his, **kwargs):
        # inference forward
        prop_latent = self.prop_his_enc(prop_his.transpose(1, 2))
        depth_latent = self.depth_enc(depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=1)

        # TODO: transformer here?
        gru_out, self.hidden_states = self.gru(gru_input.unsqueeze(0), self.hidden_states)
        return gru_out.squeeze(0)

    def forward(self, prop_his, depth_his, hidden_states, **kwargs):
        # update forward
        prop_latent = gru_wrapper(self.prop_his_enc.forward, prop_his.transpose(2, 3))

        depth_latent = gru_wrapper(self.depth_enc.forward, depth_his)

        gru_input = torch.cat((prop_latent, depth_latent), dim=2)

        gru_out, _ = self.gru(gru_input, hidden_states)
        return gru_out

    def get_hidden_states(self):
        if self.hidden_states is None:
            return None
        return self.hidden_states.detach()

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

    def reset(self, dones):
        if self.hidden_states is not None:
            self.hidden_states[:, dones] = 0.


class PIEPolicy(BaseRecurrentActor):
    is_recurrent = True

    def __init__(
            self,
            prop_shape: tuple[int, ...],
            action_size: int,
            activation=nn.ELU(),
    ):
        super().__init__(action_size=action_size)

        self.estimator = EstimatorGRU(prop_shape)

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
