from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch
import torch.optim as optim
from torch import GradScaler

from bridge_env.envs import BridgeEnv
from bridge_rl.algorithms import PPO, PPOCritic, masked_MSE
from bridge_rl.storage import RolloutStorage
from .networks import DreamWaQRecurrentActor

if TYPE_CHECKING:
    from . import DreamWaQCfg


class DreamWaQ(PPO):
    def __init__(self, cfg: DreamWaQCfg, env: BridgeEnv, **kwargs):
        super().__init__(cfg, env, **kwargs)
        self.cfg = cfg

        # Store additional config
        self.use_recurrent = cfg.use_recurrent_policy

        # Loss coefficients
        self.vel_est_loss_coef = cfg.vel_est_loss_coef
        self.ot1_pred_loss_coef = cfg.ot1_pred_loss_coef
        self.kl_coef_vel = cfg.kl_coef_vel
        self.kl_coef_z = cfg.kl_coef_z

    def _init_components(self):
        # Initialize DreamWAQ actor
        proprio_shape = self.env.observation_manager.group_obs_dim['proprio']
        action_size = self.env.action_manager.total_action_dim

        if self.cfg.use_recurrent_policy:
            self.actor = DreamWaQRecurrentActor(
                prop_shape=proprio_shape,
                vae_latent_size=self.cfg.vae_latent_size,
                num_gru_layers=self.cfg.num_gru_layers,
                gru_hidden_size=self.cfg.gru_hidden_size,
                actor_hidden_dims=self.cfg.actor_hidden_dims,
                action_size=action_size,
            ).to(self.device)
            self.actor.init_hidden_states(self.env.num_envs, device=self.device)
        else:
            raise NotImplementedError

        self.actor.reset_std(self.cfg.init_noise_std, self.device)

        # Initialize DreamWaQ critic
        critic_obs_shape = self.env.observation_manager.group_obs_dim['critic_obs']
        scan_shape = self.env.observation_manager.group_obs_dim['scan']

        self.critic = PPOCritic(
            critic_obs_shape=critic_obs_shape,
            scan_shape=scan_shape,
            critic_hidden_dims=self.cfg.critic_hidden_dims
        ).to(self.device)

        # Initialize DreamWaQ optimizer
        self.optimizer_ppo = optim.Adam([*self.actor.actor_backbone.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.optimizer_vae = optim.Adam([*self.actor.gru.parameters(), *self.actor.vae.parameters()], lr=self.learning_rate)
        self.scaler_vae = GradScaler(enabled=self.cfg.use_amp)

        # Initialize DreamWaQ storage
        self.storage = RolloutStorage(
            obs_shape=self.env.observation_manager.group_obs_dim,
            num_actions=action_size,
            storage_length=self.cfg.num_steps_per_update,
            num_envs=self.num_envs,
            device=self.device
        )

        self.storage.register_hidden_state_buffer(
            'hidden_states', self.cfg.num_gru_layers, self.cfg.gru_hidden_size
        )
        self.storage.register_data_buffer(
            'vel_est', (3,), torch.float
        )
        self.storage.register_data_buffer(
            'z_est', (self.cfg.vae_latent_size,), torch.float
        )
        self.storage.register_data_buffer(
            'prop_next', proprio_shape, torch.float
        )

    def _generate_actions(self, observations, **kwargs):
        actions, vel, z = self.actor.act(observations, **kwargs)

        self.storage.add_transitions('vel_est', vel)
        self.storage.add_transitions('z_est', z)

        return actions

    def process_env_step(self, rewards, terminated, timeouts, infos, **kwargs):
        if self.actor.is_recurrent:
            self.actor.reset(terminated | timeouts)

        self.storage.add_transitions('prop_next', kwargs['obs_next']['proprio'])

        super().process_env_step(rewards, terminated, timeouts, infos, **kwargs)

    def update(self, **kwargs) -> Dict[str, float]:
        """Main update loop for both PPO and VAE components."""
        # Clear statistics from previous update
        self.stats_tracker.clear()

        # Get batch generator
        if self.actor.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.cfg.num_mini_batches, self.cfg.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.cfg.num_mini_batches, self.cfg.num_learning_epochs
            )

        # Training loop
        for batch in generator:
            # Extract batch data
            observations = batch['observations']

            # Forward pass through actor
            self.actor.actor_forward(observations['proprio'], batch['vel_est'], batch['z_est'])

            # Update PPO
            self._update_ppo(batch)

            # # Update VAE
            # self._update_vae(batch)

        self.storage.clear()

        # Get all mean statistics
        stats = self.stats_tracker.get_means()

        # Add additional non-loss metrics
        stats.update({
            'Loss/learning_rate': self.learning_rate,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        })

        return stats

    def _update_vae(self, batch):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            observations = batch['observations']
            prop_next = batch['prop_next']
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            masks = batch['masks'] if self.actor.is_recurrent else slice(None)

            # Compute VAE forward pass using the vae_input (with recurrent wrapper)
            # VAE returns: z, vel, ot1, mu_z, logvar_z, mu_vel, logvar_vel
            vel, z, ot1, mu_vel, logvar_vel, mu_z, logvar_z = self.actor.vae_forward(observations, hidden_states)

            # Velocity estimation loss using est_gt observation
            vel_est_loss = masked_MSE(vel, observations['est_gt'], masks)

            # Next observation prediction loss
            ot1_pred_loss = masked_MSE(ot1, prop_next, masks)

            # VAE loss for latent variable z (KL divergence regularization)
            vae_loss_z = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            vae_loss_z = -0.5 * (vae_loss_z * masks).sum(dim=2).mean()

            # VAE loss for velocity (KL divergence regularization)
            vae_loss_vel = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
            vae_loss_vel = -0.5 * (vae_loss_vel * masks).sum(dim=2).mean()

            # Total VAE loss with configurable weights
            total_loss = (
                    self.vel_est_loss_coef * vel_est_loss +
                    self.ot1_pred_loss_coef * ot1_pred_loss +
                    self.kl_coef_vel * vae_loss_vel +
                    self.kl_coef_z * vae_loss_z
            )

        # VAE gradient step
        self.optimizer_vae.zero_grad()
        self.scaler_vae.scale(total_loss).backward()
        self.scaler_vae.step(self.optimizer_vae)
        self.scaler_vae.update()

        # Track VAE statistics with proper labels
        self.stats_tracker.add("VAE/vel_est_loss", vel_est_loss)
        self.stats_tracker.add("VAE/ot1_pred_loss", ot1_pred_loss)
        self.stats_tracker.add("VAE/kl_z", vae_loss_z)
        self.stats_tracker.add("VAE/kl_vel", vae_loss_vel)

        # Track VAE mu and std statistics for both z and velocity
        self.stats_tracker.add("VAE/abs_z", mu_z.abs().mean())
        self.stats_tracker.add("VAE/std_z", logvar_z.exp().sqrt().mean())
        self.stats_tracker.add("VAE/abs_vel", mu_vel.abs().mean())
        self.stats_tracker.add("VAE/std_vel", logvar_vel.exp().sqrt().mean())

    def play_act(self, obs, **kwargs):
        """Generate actions for play/evaluation."""
        return {'joint_pos': self.actor.act(obs, eval_=True, **kwargs)}

    def save(self) -> Dict[str, Any]:
        """Save model state."""
        save_dict = super().save()

        # Add DreamWAQ-specific state
        save_dict.update({
            'use_recurrent': self.use_recurrent,
            'optimizer_vae_state_dict': self.optimizer_vae.state_dict(),
        })

        return save_dict

    def load(self, loaded_dict: Dict[str, Any], load_optimizer: bool = True) -> Any:
        """Load model state."""
        # Load base PPO state
        result = super().load(loaded_dict, load_optimizer)

        # Load DreamWAQ-specific state if needed
        if 'use_recurrent' in loaded_dict:
            self.use_recurrent = loaded_dict['use_recurrent']

        # Load VAE optimizer if available
        if load_optimizer and 'optimizer_vae_state_dict' in loaded_dict:
            self.optimizer_vae.load_state_dict(loaded_dict['optimizer_vae_state_dict'])

        # Reset noise if specified
        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, self.device)

        return result
