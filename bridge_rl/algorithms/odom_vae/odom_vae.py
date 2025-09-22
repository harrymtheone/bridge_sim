from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch.optim as optim

from bridge_env.envs import BridgeEnv
from bridge_rl.algorithms import PPO, PPOCritic, recurrent_wrapper
from bridge_rl.storage import RolloutStorage
from .networks import OdomVAEActor

if TYPE_CHECKING:
    from .odom_vae_cfg import OdomCfg


class OdomVAE(PPO):
    def __init__(self, cfg: OdomCfg, env: BridgeEnv, **kwargs):
        super().__init__(cfg, env, **kwargs)
        self.cfg = cfg

    def _init_components(self):
        # derive shapes from env
        prop_shape = self.env.observation_manager.group_obs_dim['proprio']
        # est_shape = self.env.observation_manager.group_obs_dim['est_gt']
        scan_shape = self.env.observation_manager.group_obs_dim['scan']
        critic_obs_shape = self.env.observation_manager.group_obs_dim['critic_obs']
        action_size = self.env.action_manager.total_action_dim

        # initialize networks
        self.actor = OdomVAEActor(
            prop_shape=prop_shape,
            scan_shape=scan_shape,
            vae_latent_size=self.cfg.vae_latent_size,
            actor_gru_hidden_size=self.cfg.actor_gru_hidden_size,
            actor_gru_num_layers=self.cfg.actor_gru_num_layers,
            actor_hidden_dims=self.cfg.actor_hidden_dims,
            action_size=action_size,
        ).to(self.device)
        self.actor.init_hidden_states(self.env.num_envs, self.device)

        self.critic = PPOCritic(
            critic_obs_shape=critic_obs_shape,
            scan_shape=scan_shape,
            critic_hidden_dims=self.cfg.critic_hidden_dims
        ).to(self.device)

        # optimizer
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        self.optimizer_vae = optim.Adam(self.actor.vae.parameters(), lr=self.learning_rate)

        # storage
        self.storage = RolloutStorage(
            obs_shape=self.env.observation_manager.group_obs_dim,
            num_actions=action_size,
            storage_length=self.cfg.num_steps_per_update,
            num_envs=self.env.num_envs,
            device=self.device,
        )

        # hidden state buffer for recurrent actor
        self.storage.register_hidden_state_buffer(
            'hidden_states',
            num_layers=self.cfg.actor_gru_num_layers,
            hidden_size=self.cfg.actor_gru_hidden_size
        )

    def _generate_actions(self, observations, **kwargs):
        return self.actor.act(observations, **kwargs)

    def process_env_step(self, rewards, terminated, timeouts, infos, **kwargs):
        if self.actor.is_recurrent:
            self.actor.reset(terminated | timeouts)

        super().process_env_step(rewards, terminated, timeouts, infos, **kwargs)

    def update(self, **kwargs) -> Dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_kl = 0

        kl_change = []
        num_updates = 0

        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        for batch in generator:
            # Extract batch data for actor forward pass
            observations = batch['observations']
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            masks = batch['masks'].squeeze(-1) if self.actor.is_recurrent else slice(None)

            # Forward pass through actor
            vae_input = self.actor.train_act(observations, hidden_states=hidden_states)

            # Compute PPO loss
            kl_mean, surrogate_loss, value_loss, entropy_loss = self._compute_ppo_loss(batch)

            # Track KL divergence for adaptive learning rate
            kl_change.append(kl_mean)
            num_updates += 1
            mean_kl += kl_mean
            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_entropy_loss += entropy_loss.item()

            # Combine losses
            total_loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy_loss

            # Gradient step
            self._update_learning_rate(kl_mean)
            self._gradient_step(total_loss)

            if self.cfg.noise_std_range:
                self.actor.clip_std(self.cfg.noise_std_range[0], self.cfg.noise_std_range[1])

            # compute VAE loss
            z, vel, ot1, mu_z, logvar_z, mu_vel, logvar_vel = recurrent_wrapper(self.actor.vae.forward, vae_input.detach())

            # velocity estimation loss
            estimation_loss = self.mse_loss(vel[masks], observations['est_gt'][masks])

            # Ot+1 prediction loss  TODO: what???
            # prediction_loss = self.mse_loss(ot1[masks], obs_next_batch.proprio[masks])

            # VAE loss
            vae_loss_z = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            vae_loss_z = -0.5 * vae_loss_z[masks].sum(dim=1).mean()

            vae_loss_vel = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
            vae_loss_vel = -0.5 * vae_loss_vel[masks].sum(dim=1).mean()

            loss = estimation_loss + vae_loss_z + vae_loss_vel

            self.optimizer_vae.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Average statistics
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_kl /= num_updates

        # Print KL divergence tracking
        kl_str = 'kl: '
        for k in kl_change:
            kl_str += f'{k:.3f} | '
        print(kl_str)

        self.storage.clear()

        # Base statistics
        stats = {
            'Loss/learning_rate': self.learning_rate,
            'Loss/value_loss': mean_value_loss,
            'Loss/kl_div': mean_kl,
            'Loss/surrogate_loss': mean_surrogate_loss,
            'Loss/entropy_loss': mean_entropy_loss,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        }

        return stats

    def play_act(self, obs, eval_=True, **kwargs):
        """Generate actions for play/evaluation."""
        return {'actions': self.actor.act(obs, eval_=eval_, **kwargs)}
