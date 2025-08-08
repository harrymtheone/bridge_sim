from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from bridge_rl.algorithms import make_linear_layers, recurrent_wrapper, BaseRecurrentActor


class OdomActor(BaseRecurrentActor):
    is_recurrent = True

    def __init__(
            self,
            prop_shape: tuple[int, ...],
            priv_shape: tuple[int, ...],
            scan_shape: tuple[int, ...],
            actor_gru_hidden_size: int,
            actor_hidden_dims: tuple[int, ...],
            action_size: int,
    ):
        super().__init__(action_size=action_size)
        activation = nn.ELU()

        assert len(prop_shape) == 1 and len(priv_shape) == 1 and len(scan_shape) == 1
        prop_size, priv_size, scan_size = prop_shape[0], priv_shape[0], scan_shape[0]

        # Encodes height scan or alternative reconstructed scan
        self.scan_encoder = make_linear_layers(
            scan_size, 256, 128,
            activation_func=activation
        )

        # Belief encoder (GRU)
        self.gru = nn.GRU(prop_size + 128 + priv_size, actor_gru_hidden_size, num_layers=1)

        # Actor MLP head
        self.actor = make_linear_layers(
            actor_gru_hidden_size,
            *actor_hidden_dims,
            action_size,
            activation_func=activation,
            output_activation=False,
        )

    def act(self, obs, eval_: bool = False, **kwargs):
        proprio = obs['proprio']
        scan = obs['scan']
        priv = obs['priv']
        use_estimated_values = obs['use_estimated_values']

        if torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                self.scan_encoder(hmap.flatten(1)),
                self.scan_encoder(obs.scan.flatten(1)),
            )
            x = torch.where(
                use_estimated_values,
                torch.cat([proprio, scan_enc, priv], dim=1),
                torch.cat([proprio, scan_enc, obs.priv_actor], dim=1),
            )
        else:
            scan_enc = self.scan_encoder(scan.flatten(1))
            x = torch.cat([proprio, scan_enc, priv], dim=1)

        # GRU forward
        x, self.hidden_states = self.gru(x.unsqueeze(0), self.hidden_states)
        x = x.squeeze(0)

        mean = self.actor(x)
        if eval_:
            return mean
        self.distribution = Normal(mean, torch.exp(self.log_std))
        return self.distribution.sample()

    def train_act(self, obs, hidden_states=None, **kwargs):
        proprio = obs['proprio']
        scan = obs['scan']
        priv = obs['priv']
        use_estimated_values = obs['use_estimated_values']

        if use_estimated_values is not None and torch.any(use_estimated_values):
            scan_enc = torch.where(
                use_estimated_values,
                recurrent_wrapper(self.scan_encoder.forward, scan.flatten(2)),
                recurrent_wrapper(self.scan_encoder.forward, obs.scan.flatten(2)),
            )
            x = torch.where(
                use_estimated_values,
                torch.cat([obs.proprio, scan_enc, est], dim=2),
                torch.cat([obs.proprio, scan_enc, obs.priv_actor], dim=2),
            )
        else:
            scan_enc = recurrent_wrapper(self.scan_encoder.forward, scan.flatten(2))
            x = torch.cat([proprio, scan_enc, priv], dim=2)

        x, _ = self.gru(x, hidden_states)
        mean = recurrent_wrapper(self.actor.forward, x)
        self.distribution = Normal(mean, torch.exp(self.log_std))

    def init_hidden_states(self, num_envs, device):
        self.hidden_states = torch.zeros(1, num_envs, self.gru.hidden_size, device=device)
