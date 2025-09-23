from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

import torch
from torch.distributions import kl_divergence, Normal

from bridge_env.envs import BridgeEnv
from bridge_rl.storage import RolloutStorage
from bridge_rl.utils import StatisticsTracker

if TYPE_CHECKING:
    from . import PPOCfg, BaseActor, BaseCritic


class PPO:
    def __init__(self, cfg: PPOCfg, env: BridgeEnv, **kwargs):
        cfg.validate()  # noqa

        self.cfg = cfg
        self.env = env
        self.device = torch.device(env.device)
        self.learning_rate = self.cfg.learning_rate

        self.actor: BaseActor | None = None
        self.critic: BaseCritic | None = None
        self.optimizer = None
        self.storage: RolloutStorage | None = None

        # Mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=self.cfg.use_amp)

        # Common loss functions
        self.mse_loss = torch.nn.MSELoss()

        # Initialize components
        self._init_components()

        self.stats_tracker = StatisticsTracker()

    def _init_components(self):
        raise NotImplementedError

    def act(self, observations, **kwargs):
        # Store observations
        self._store_observations(observations)

        # Handle hidden states
        self._store_hidden_states()

        # Generate actions
        actions = self._generate_actions(observations, **kwargs)

        # Evaluate using critic
        values = self.critic.evaluate(observations)

        # Store transition data
        self._store_transition_data(actions, values, **kwargs)

        return {'joint_pos': actions}  # TODO: after add support for single action

    def _store_observations(self, observations: dict[str, torch.Tensor]):
        self.storage.add_transitions('observations', observations)

    def _store_hidden_states(self):
        if self.actor.is_recurrent:
            self.storage.add_transitions('hidden_states', self.actor.get_hidden_states())

    def _generate_actions(self, obs, **kwargs):
        raise NotImplementedError

    def _store_transition_data(self, actions, values, **kwargs):
        self.storage.add_transitions('values', values)
        self.storage.add_transitions('actions', actions)
        self.storage.add_transitions('actions_log_prob', self.actor.get_actions_log_prob(actions))
        self.storage.add_transitions('action_mean', self.actor.action_mean)
        self.storage.add_transitions('action_sigma', self.actor.action_std)

    def process_env_step(self, rewards, terminated, timeouts, infos, **kwargs):
        self.storage.add_transitions('rewards', rewards.unsqueeze(1))
        self.storage.add_transitions('dones', (terminated | timeouts).unsqueeze(1))

        self.storage.flush()

    def compute_returns(self, last_observations):
        last_values = self.critic.evaluate(last_observations).detach()
        self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

    def update(self, **kwargs) -> Dict[str, float]:
        raise NotImplementedError

    def _compute_ppo_loss(self, batch: Dict[str, torch.Tensor]):
        with torch.autocast(str(self.device), torch.float16, enabled=self.cfg.use_amp):
            # Extract batch data
            observations = batch['observations']
            masks = batch['masks'].squeeze(-1) if self.actor.is_recurrent else slice(None)
            actions = batch['actions']
            target_values = batch['values']
            advantages = batch['advantages']
            returns = batch['returns']
            old_mu = batch['action_mean']
            old_sigma = batch['action_sigma']
            old_actions_log_prob = batch['actions_log_prob']

            # Compute KL divergence for adaptive learning rate
            if self.cfg.learning_rate_schedule == 'adaptive':
                with torch.no_grad():
                    kl_mean = kl_divergence(
                        Normal(old_mu, old_sigma),
                        Normal(self.actor.action_mean, self.actor.action_std)
                    )[masks].sum(dim=-1).mean().item()

            # Compute action log probabilities
            actions_log_prob = self.actor.get_actions_log_prob(actions)
            evaluation = self.critic.evaluate(observations)

            # Surrogate loss (PPO clipped objective)
            ratio = torch.exp(actions_log_prob - old_actions_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped)[masks].mean()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = target_values + (evaluation - target_values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (returns - evaluation).pow(2)
                value_losses_clipped = (returns - value_clipped).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped)[masks].mean()
            else:
                value_loss = (returns - evaluation)[masks].pow(2).mean()

            # Entropy loss
            entropy_loss = self.actor.entropy[masks].mean()

        return kl_mean, surrogate_loss, value_loss, entropy_loss

    def _update_ppo(self, batch: dict[str, torch.Tensor]):
        """Update PPO policy and critic networks.

        Args:
            batch: Mini-batch data from storage

        Returns:
            KL divergence mean for printing
        """
        # Compute PPO loss
        kl_mean, surrogate_loss, value_loss, entropy_loss = self._compute_ppo_loss(batch)

        # Combine losses
        loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy_loss

        # Gradient step
        self._update_learning_rate(kl_mean)
        self._gradient_step(loss)

        if self.cfg.noise_std_range:
            self.actor.clip_std(self.cfg.noise_std_range[0], self.cfg.noise_std_range[1])

        # Track PPO statistics with proper labels
        self.stats_tracker.add("Loss/kl_div", kl_mean)
        self.stats_tracker.add("Loss/surrogate_loss", surrogate_loss)
        self.stats_tracker.add("Loss/value_loss", value_loss)
        self.stats_tracker.add("Loss/entropy_loss", entropy_loss)

    def _update_learning_rate(self, kl_mean: float):
        if self.cfg.learning_rate_schedule == 'adaptive' and self.cfg.desired_kl is not None:
            if kl_mean > self.cfg.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif self.cfg.desired_kl / 2.0 > kl_mean > 0.0:
                self.learning_rate = min(1e-3, self.learning_rate * 1.5)

            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def _gradient_step(self, loss: torch.Tensor):
        """Perform gradient step with optional clipping.
        
        Args:
            loss: Total loss tensor
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        if self.cfg.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [*self.actor.parameters(), *self.critic.parameters()],
                self.cfg.max_grad_norm
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def play_act(self, obs, eval_=True, **kwargs):
        """Generate actions for inference/evaluation.
        
        Args:
            obs: Observations
            eval_: Sample action or use mean as action
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing actions
        """
        raise NotImplementedError

    def train(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    def load(self, loaded_dict: Dict[str, Any], load_optimizer: bool = True) -> Any:
        """Load model state from checkpoint.
        
        Args:
            loaded_dict: Dictionary containing saved state
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Info from loaded dictionary
        """
        if self.actor is not None and 'actor_state_dict' in loaded_dict:
            self.actor.load_state_dict(loaded_dict['actor_state_dict'])

        if self.critic is not None and 'critic_state_dict' in loaded_dict:
            self.critic.load_state_dict(loaded_dict['critic_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        # Reset noise standard deviation if needed
        if not self.cfg.continue_from_last_std:
            self.actor.reset_std(self.cfg.init_noise_std, device=self.device)

        return loaded_dict.get('infos', None)

    def save(self) -> Dict[str, Any]:
        """Save model state to dictionary.
        
        Returns:
            Dictionary containing model state
        """
        save_dict = {}

        if self.actor is not None:
            save_dict['actor_state_dict'] = self.actor.state_dict()

        if self.critic is not None:
            save_dict['critic_state_dict'] = self.critic.state_dict()

        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        return save_dict

    def reset(self, dones):
        """Reset algorithm state for environments that are done.
        
        Args:
            dones: Boolean tensor indicating which environments are done
        """
        if self.actor.is_recurrent:
            self.actor.reset(dones)
