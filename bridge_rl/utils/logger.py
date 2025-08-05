import torch
from typing import Dict, List, Optional
from collections import deque
import statistics


class EpisodeLogger:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        
        self.cur_episode_length = torch.zeros(num_envs, dtype=torch.int32)
        self.episode_length = deque(maxlen=100)
        
        self.cur_episode_reward_sum = torch.zeros(num_envs, dtype=torch.float32)
        self.episode_reward_sum = deque(maxlen=100)
        
    def reset(self):
        self.cur_episode_length.zero_()
        self.cur_episode_reward_sum.zero_()
        
    def step(self, rewards: torch.Tensor, terminated: torch.Tensor, timeouts: torch.Tensor):
        self.cur_episode_length[:] += 1
        self.cur_episode_reward_sum[:] += rewards.cpu()
        
        episode_ended = (terminated | timeouts).cpu()
        if episode_ended.any():
            completed_lengths = self.cur_episode_length[episode_ended]
            completed_rewards = self.cur_episode_reward_sum[episode_ended]
            
            self.episode_length.extend(completed_lengths.tolist())
            self.episode_reward_sum.extend(completed_rewards.tolist())
            
            self.cur_episode_length[episode_ended] = 0
            self.cur_episode_reward_sum[episode_ended] = 0
            
    def get_logging_dict(self) -> Dict[str, float]:
        if not self.episode_length:
            return {}
                       
        return {
            'Train/mean_episode_length': statistics.mean(self.episode_length),
            'Train/mean_episode_reward': statistics.mean(self.episode_reward_sum),
        }

    def get_buffer_info(self) -> Dict[str, int]:
        return {
            'buffer_size': len(self.episode_length),
            'buffer_capacity': self.episode_length.maxlen
        }