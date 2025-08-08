from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
import torch.optim as optim
from isaaclab.envs import ManagerBasedRLEnv

from bridge_rl.algorithms import PPO
from bridge_rl.storage import RolloutStorage
from .networks import DreamWaQActor, DreamWaQRecurrentActor, DreamWaQCritic

if TYPE_CHECKING:
    from . import DreamWaQCfg


class DreamWaQ(PPO):
    def __init__(self, cfg: DreamWaQCfg, env: ManagerBasedRLEnv, **kwargs):
        super().__init__(cfg, env, **kwargs)
        self.cfg = cfg

        # Store additional config
        self.use_recurrent = cfg.use_recurrent_policy
        self.update_estimation = cfg.update_estimation

        # Loss coefficients
        self.symmetry_loss_coef = cfg.symmetry_loss_coef
        self.estimation_loss_coef = cfg.estimation_loss_coef
        self.prediction_loss_coef = cfg.prediction_loss_coef
        self.vae_loss_coef = cfg.vae_loss_coef

    def _init_components(self):
        # Initialize DreamWAQ actor
        proprio_shape = self.env.observation_manager.group_obs_dim['proprio']
        action_size = self.env.action_manager.total_action_dim

        if self.cfg.use_recurrent_policy:
            self.actor = DreamWaQRecurrentActor(
                prop_shape=proprio_shape,
                num_gru_layers=self.cfg.num_gru_layers,
                gru_hidden_size=self.cfg.gru_hidden_size,
                actor_hidden_dims=self.cfg.actor_hidden_dims,
                encoder_output_size=self.cfg.encoder_output_size,
                action_size=action_size,
            ).to(self.device)
        else:
            raise NotImplementedError
            self.actor = DreamWaQActor(
                obs_size=self.cfg.actor_obs_size,  # This should be set in config
                action_size=self.cfg.action_size,  # This should be set in config
                hidden_dims=self.cfg.actor_hidden_dims,
                encoder_output_size=self.cfg.encoder_output_size
            ).to(self.device)

        self.actor.reset_std(self.cfg.init_noise_std, self.device)

        # Initialize DreamWaQ critic
        critic_obs_shape = self.env.observation_manager.group_obs_dim['critic_obs']
        scan_shape = self.env.observation_manager.group_obs_dim['scan']

        self.critic = DreamWaQCritic(
            critic_obs_shape=critic_obs_shape,  # This should be set in config
            scan_shape=scan_shape,  # This should be set in config
            hidden_dims=self.cfg.critic_hidden_dims
        ).to(self.device)

        # Initialize DreamWaQ optimizer
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Initialize DreamWaQ storage
        self.storage = RolloutStorage(
            obs_shape=self.env.observation_manager.group_obs_dim,
            num_actions=action_size,
            storage_length=24,
            num_envs=self.env.num_envs,
            device=self.device
        )

        self.storage.register_hidden_state_buffer(
            'hidden_states', self.cfg.num_gru_layers, self.cfg.gru_hidden_size
        )

    def _generate_actions(self, observations, **kwargs):
        return self.actor.act(observations, **kwargs)

    def _store_transition_data(self, actions, values, **kwargs):
        super()._store_transition_data(actions, values, **kwargs)

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
            # Compute losses
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

    def _compute_additional_losses(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Compute DreamWAQ-specific losses."""
        additional_losses = {}

        # Get batch data
        obs_batch = batch['observations']
        critic_obs_batch = batch['critic_observations']
        hidden_states_batch = batch.get('hidden_states') if self.actor.is_recurrent else None
        mask_batch = batch.get('masks', slice(None))
        if isinstance(mask_batch, torch.Tensor):
            mask_batch = mask_batch.squeeze()

        # Compute symmetry loss
        symmetry_loss = self._compute_symmetry_loss(obs_batch, hidden_states_batch, mask_batch)
        additional_losses['symmetry_loss'] = self.symmetry_loss_coef * symmetry_loss

        # Compute estimation losses if enabled
        if self.update_estimation and 'observations_next' in batch:
            estimation_losses = self._compute_estimation_losses(batch, mask_batch)
            additional_losses.update(estimation_losses)

        return additional_losses

    def _compute_symmetry_loss(self, obs_batch, hidden_states_batch, mask_batch):
        """Compute symmetry loss for robust policy learning."""
        # Get original action mean
        self.actor.train_act(obs_batch, hidden_states=hidden_states_batch)
        action_mean_original = self.actor.action_mean.detach()

        # Create mirrored observations
        obs_mirrored_batch = obs_batch.flatten(0, 1).mirror().unflatten(0, (obs_batch.size(0), -1))

        # Get mirrored action mean
        self.actor.train_act(obs_mirrored_batch, hidden_states=hidden_states_batch)

        # Compute expected mirrored actions
        mu_batch = obs_batch.mirror_dof_prop_by_x(action_mean_original.flatten(0, 1)).unflatten(0, (obs_batch.size(0), -1))

        # Compute symmetry loss
        symmetry_loss = self.mse_loss(mu_batch[mask_batch], self.actor.action_mean[mask_batch])

        return symmetry_loss

    def _compute_estimation_losses(self, batch: Dict[str, torch.Tensor], mask_batch):
        """Compute VAE and estimation losses."""
        obs_batch = batch['observations']
        obs_next_batch = batch['observations_next']
        hidden_states_batch = batch.get('hidden_states') if self.actor.is_recurrent else None

        # Get VAE estimates
        ot1, est_vel, est_mu, est_logvar = self.actor.estimate(obs_batch, hidden_states=hidden_states_batch)

        # Privileged information estimation loss
        estimation_loss = self.mse_loss(est_vel[mask_batch], obs_batch.priv_actor[mask_batch])

        # Next observation prediction loss
        prediction_loss = self.mse_loss(ot1[mask_batch], obs_next_batch.proprio[mask_batch])

        # VAE loss (KL divergence regularization)
        vae_loss = 1 + est_logvar - est_mu.pow(2) - est_logvar.exp()
        vae_loss = -0.5 * vae_loss[mask_batch].sum(dim=1).mean()

        return {
            'estimation_loss': self.estimation_loss_coef * estimation_loss,
            'prediction_loss': self.prediction_loss_coef * prediction_loss,
            'vae_loss': self.vae_loss_coef * vae_loss
        }

    def _get_additional_stats(self) -> Optional[Dict[str, float]]:
        """Get additional training statistics."""
        stats = {}

        # Add noise standard deviation
        if hasattr(self.actor, 'log_std'):
            stats['Train/noise_std'] = self.actor.log_std.exp().mean().item()

        return stats

    def play_act(self, obs, **kwargs):
        """Generate actions for play/evaluation."""
        return {'actions': self.actor.act(obs, eval_=True, **kwargs)}

    def save(self) -> Dict[str, Any]:
        """Save model state."""
        save_dict = super().save()

        # Add DreamWAQ-specific state if needed
        save_dict.update({
            'use_recurrent': self.use_recurrent,
            'update_estimation': self.update_estimation,
        })

        return save_dict

    def load(self, loaded_dict: Dict[str, Any], load_optimizer: bool = True) -> Any:
        """Load model state."""
        # Load base PPO state
        result = super().load(loaded_dict, load_optimizer)

        # Load DreamWAQ-specific state if needed
        if 'use_recurrent' in loaded_dict:
            self.use_recurrent = loaded_dict['use_recurrent']
        if 'update_estimation' in loaded_dict:
            self.update_estimation = loaded_dict['update_estimation']

        # Reset noise if specified
        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, self.device)

        return result
