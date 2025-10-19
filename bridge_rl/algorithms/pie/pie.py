from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch
import torch.optim as optim

from bridge_env.envs import BridgeEnv
from bridge_rl.algorithms import PPO, PPOCritic, recurrent_wrapper
from bridge_rl.storage import RolloutStorage
from .networks import Mixer, EstimatorVAE, Actor

if TYPE_CHECKING:
    from .pie_cfg import PIECfg


class PIE(PPO):
    cfg: PIECfg

    def __init__(self, cfg: PIECfg, env: BridgeEnv, **kwargs):
        super().__init__(cfg, env, **kwargs)

    def _init_components(self):
        # derive shapes from env
        prop_shape = self.env.observation_manager.group_obs_dim['proprio']
        scan_shape = self.env.observation_manager.group_obs_dim['scan']
        critic_obs_shape = self.env.observation_manager.group_obs_dim['critic_obs']
        action_size = self.env.action_manager.total_action_dim

        if len(prop_shape) == 1:
            prop_size = prop_shape[0]
        else:
            raise ValueError(f'Only support proprio without history')

        if len(scan_shape) == 1:
            scan_size = scan_shape[0]
        else:
            raise ValueError(f'Only support 1D scan input')

        # initialize networks
        self.mixer = Mixer(
            prop_size=prop_size,
            depth_channel=self.cfg.mixer_depth_channel,
            hidden_size=self.cfg.mixer_hidden_size,
        ).to(self.device)
        self.mixer_hidden_states = self.mixer.init_hidden_states(self.num_envs, self.device)

        ot1_size = prop_size - self.cfg.len_command - action_size
        self.vae = EstimatorVAE(
            input_size=self.cfg.mixer_hidden_size,
            len_latent_z=self.cfg.len_latent_z,
            len_latent_hmap=self.cfg.len_latent_hmap,
            ot1_size=ot1_size,
            scan_size=scan_size,
        ).to(self.device)

        self.actor = Actor(
            prop_size=prop_size,
            len_latent=self.cfg.len_latent_z + self.cfg.len_latent_hmap,
            actor_hidden_dims=self.cfg.actor_hidden_dims,
            action_size=action_size,
        ).to(self.device)

        self.critic = PPOCritic(
            critic_obs_shape=critic_obs_shape,
            scan_shape=scan_shape,
            critic_hidden_dims=self.cfg.critic_hidden_dims
        ).to(self.device)

        # optimizers
        self.optimizer_ppo = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=self.learning_rate)
        self.optimizer_vae = optim.Adam([*self.mixer.parameters(), *self.vae.parameters()], lr=self.learning_rate)

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
            'mixer_hidden_states',
            num_layers=1,
            hidden_size=self.cfg.mixer_hidden_size
        )
        self.storage.register_data_buffer(
            'actor_input',
            data_shape=(prop_size + 3 + self.cfg.len_latent_z + self.cfg.len_latent_hmap,),
            dtype=torch.float,
        )

        # next proprio buffer for VAE
        self.storage.register_data_buffer('prop_next', ot1_size, torch.float)

    def act(self, observations, **kwargs) -> dict[str, torch.Tensor]:
        observations = self.preprocess_observations(observations)

        # Store observations
        self.storage.add_transitions('observations', observations)
        self.storage.add_transitions('mixer_hidden_states', self.mixer_hidden_states)

        proprio = observations['proprio']
        depth = observations['depth']

        mixer_out, self.mixer_hidden_states[:] = self.mixer(proprio, depth, self.mixer_hidden_states)
        vel, z, mu_vel, logvar_vel, mu_z, logvar_z, ot1, hmap = self.vae(mixer_out, sample=True)

        actor_input = torch.cat([proprio, vel, z], dim=-1)
        self.storage.add_transitions('actor_input', actor_input)
        actions = self.actor.sample_actions(actor_input).squeeze(0)

        values = self._evaluate_state(observations)

        # Store transition data
        self._store_transition_data(actions, values, **kwargs)

        return {'joint_pos': actions}

    def preprocess_observations(self, observations):
        obs_processed = {}
        for k, v in observations.items():
            obs_processed[k] = v.unsqueeze(0)

        return obs_processed

    def _evaluate_state(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        critic_obs = obs["critic_obs"]
        scan = obs["scan"]
        return self.critic(critic_obs, scan)

    def process_env_step(self, rewards, terminated, timeouts, infos, **kwargs):
        self.mixer_hidden_states[:, terminated | timeouts] = 0.

        if 'obs_next' in kwargs:
            obs_next = kwargs['obs_next']
            self.storage.add_transitions('prop_next', obs_next['prop_next'])
        else:
            raise RuntimeError("ObsNext should be provided for Ot+1 estimation")

        super().process_env_step(rewards, terminated, timeouts, infos, **kwargs)

    def update(self, **kwargs) -> Dict[str, float]:
        """Main update loop for both PPO and VAE components."""
        # Clear statistics from previous update
        self.stats_tracker.clear()

        # Training loop
        for batch in self.storage.recurrent_mini_batch_generator(
                self.cfg.num_mini_batches, self.cfg.num_learning_epochs
        ):
            # Extract batch data
            observations = batch['observations']
            prop_next = batch['prop_next']
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            masks = batch['masks']

            # Forward pass through actor to get VAE input
            vae_input = self.actor.train_act(observations, hidden_states=hidden_states)

            # Update PPO (statistics tracked internally)
            self._update_ppo(batch)

            # Update VAE (statistics tracked internally)
            self._update_vae(vae_input.detach(), observations, prop_next, masks)

        self.storage.clear()

        # Get all mean statistics (already have proper labels)
        stats = self.stats_tracker.get_means()

        # Add additional non-loss metrics
        stats.update({
            'Loss/learning_rate': self.learning_rate,
            'Train/noise_std': self.actor.log_std.exp().mean().item(),
        })

        return stats

    def _update_vae(self, vae_input, observations, prop_next, masks):
        """Update VAE network.

        Args:
            vae_input: Input to VAE network
            observations: Batch observations
            masks: Masks for recurrent networks
        """
        # Compute VAE forward pass
        z, vel, ot1, mu_z, logvar_z, mu_vel, logvar_vel = recurrent_wrapper(
            self.actor.vae.forward, vae_input.detach()
        )

        # Velocity estimation loss
        vel_est_loss = self.mse_loss(vel[masks], observations['est_gt'][masks])

        # Ot+1 prediction loss
        ot1_pred_loss = self.mse_loss(ot1[masks], prop_next[masks])

        # VAE loss for latent variable z
        vae_loss_z = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
        vae_loss_z = -0.5 * vae_loss_z[masks].sum(dim=1).mean()

        # VAE loss for velocity
        vae_loss_vel = 1 + logvar_vel - mu_vel.pow(2) - logvar_vel.exp()
        vae_loss_vel = -0.5 * vae_loss_vel[masks].sum(dim=1).mean()

        # Total VAE loss with configurable weights
        loss = vel_est_loss + ot1_pred_loss + self.cfg.vae_loss_z_coef * vae_loss_z + self.cfg.vae_loss_vel_coef * vae_loss_vel

        # VAE gradient step
        self.optimizer_vae.zero_grad()
        loss.backward()
        self.optimizer_vae.step()

        # Track VAE statistics with proper labels
        self.stats_tracker.add("Loss/vae_loss_z", vae_loss_z)
        self.stats_tracker.add("Loss/vae_loss_vel", vae_loss_vel)
        self.stats_tracker.add("Loss/vel_est_loss", vel_est_loss)
        self.stats_tracker.add("Loss/ot1_pred_loss", ot1_pred_loss)

        # Track VAE mu and std statistics
        self.stats_tracker.add("VAE/mu_vel", mu_vel.mean())
        self.stats_tracker.add("VAE/mu_z", mu_z.mean())
        self.stats_tracker.add("VAE/std_vel", logvar_vel.exp().sqrt().mean())
        self.stats_tracker.add("VAE/std_z", logvar_z.exp().sqrt().mean())

    def play_act(self, obs, eval_=True, **kwargs):
        """Generate actions for play/evaluation."""
        return {'joint_pos': self.actor.act(obs, eval_=eval_, **kwargs)}

    def load(self, loaded_dict: Dict[str, Any], load_optimizer: bool = True) -> Any:
        super().load(loaded_dict, load_optimizer)

        if load_optimizer and 'optimizer_vae_state_dict' in loaded_dict:
            self.optimizer_vae.load_state_dict(loaded_dict['optimizer_vae_state_dict'])

    def save(self) -> Dict[str, Any]:
        save_dict = super().save()
        save_dict['optimizer_vae_state_dict'] = self.optimizer_vae.state_dict()
        return save_dict
