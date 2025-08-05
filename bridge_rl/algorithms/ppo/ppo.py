from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
from isaaclab.envs import ManagerBasedRLEnv
from torch.distributions import kl_divergence, Normal

from ...storage import RolloutStorage

if TYPE_CHECKING:
    from . import PPOCfg, BaseActor, BaseCritic


class PPO:
    def __init__(self, cfg: PPOCfg, env: ManagerBasedRLEnv, **kwargs):
        cfg.validate()

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

    def _init_components(self):
        raise NotImplementedError

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

        return actions

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
            hidden_states = batch['hidden_states'] if self.actor.is_recurrent else None
            masks = batch['masks'].squeeze(-1) if self.actor.is_recurrent else slice(None)
            actions = batch['actions']
            target_values = batch['values']
            advantages = batch['advantages']
            returns = batch['returns']
            old_mu = batch['action_mean']
            old_sigma = batch['action_sigma']
            old_actions_log_prob = batch['actions_log_prob']

            # Forward pass through actor
            self.actor.train_act(observations, hidden_states=hidden_states)

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
                value_losses = (evaluation - returns).pow(2)
                value_losses_clipped = (value_clipped - returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped)[masks].mean()
            else:
                value_loss = (evaluation - returns)[masks].pow(2).mean()

            # Entropy loss
            entropy_loss = self.cfg.entropy_coef * self.actor.entropy[masks].mean()

        return kl_mean, surrogate_loss, value_loss, entropy_loss

    def _prepare_hidden_states_batch(self, main_hidden_states, additional_hidden_states):
        """Prepare hidden states batch for actor. Override for custom hidden state handling."""
        # Default behavior: return tuple of all hidden states
        all_hidden_states = [main_hidden_states]
        for key in sorted(additional_hidden_states.keys()):
            all_hidden_states.append(additional_hidden_states[key])
        return tuple(all_hidden_states) if len(all_hidden_states) > 1 else main_hidden_states

    def _compute_additional_losses(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """Compute additional losses specific to subclasses.
        
        Args:
            batch: Mini-batch data
            
        Returns:
            Dictionary of additional losses or None
        """
        return None

    def _get_additional_stats(self) -> Optional[Dict[str, float]]:
        """Get additional training statistics from subclasses.
        
        Returns:
            Dictionary of additional statistics or None
        """
        return None

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

    def play_act(self, obs, **kwargs):
        """Generate actions for inference/evaluation.
        
        Args:
            obs: Observations
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
        if (hasattr(self.actor, 'reset_std') and
                not self.cfg.continue_from_last_std):
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
        if hasattr(self.actor, 'is_recurrent') and self.actor.is_recurrent:
            self.actor.reset(dones)
