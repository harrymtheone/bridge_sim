from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

import torch
import wandb
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from . import RLRunnerCfg


class RLRunner:
    def __init__(self, cfg: RLRunnerCfg, env: ManagerBasedRLEnv, device: torch.device):
        cfg.validate()

        self.cfg = cfg
        self.device = device
        self.env = env

        # Create algorithm
        self.algorithm = cfg.algorithm_cfg.class_type(self.cfg.algorithm_cfg, env=env)

        self.start_it = 0

    def learn(self):
        self.algorithm.train()  # switch to train mode (for dropout for example)

        observations, info = self.env.reset()

        for self.cur_it in range(self.start_it, self.start_it + self.cfg.max_iterations):

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_env):
                    actions = self.algorithm.act(observations)
                    observations, rewards, terminated, timeouts, infos = self.env.step(actions)
                    self.algorithm.process_env_step(rewards, terminated, timeouts, infos)

                # Learning step
                self.algorithm.compute_returns(observations)

            update_info = self.algorithm.update()

            # self.log(update_info)
            #
            # if self.cur_it % self.save_interval == 0:
            #     self.save(os.path.join(self.model_dir, f'model_{self.cur_it}.pt'))
            # self.save(os.path.join(self.model_dir, 'latest.pt'))

    def log(self, update_info, width=80, pad=35):
        self.tot_steps += self.cfg.num_steps_per_env * self.cfg.env.num_envs
        iteration_time = self.collection_time + self.learn_time
        self.tot_time += iteration_time

        # construct wandb logging dict
        logger_dict = {}

        # logging episode reward
        ep_rew = self.episode_rew
        for rew_name in ep_rew[0]:
            rew_tensor = [ep[rew_name] for ep in ep_rew]
            rew_tensor = torch.stack(rew_tensor, dim=0)
            logger_dict['Episode_rew/' + rew_name] = torch.mean(rew_tensor).item()

        # logging episode average terrain level
        ep_terrain_level = self.episode_terrain_level
        if len(ep_terrain_level) > 0:
            for terrain_name in ep_terrain_level[0]:
                level_tensor = [ep[terrain_name] for ep in ep_terrain_level]
                level_tensor = torch.stack(level_tensor, dim=0)
                logger_dict['Terrain Level/' + terrain_name] = torch.mean(level_tensor).item()

        logger_dict.update(self.terrain_coefficient_variation)

        # logging update information
        logger_dict.update(update_info)

        if len(self.episode_rew_sum) > 10:
            logger_dict['Train/mean_reward'] = statistics.mean(self.episode_rew_sum)  # use the latest 100 to compute
            logger_dict['Train/mean_episode_length'] = statistics.mean(self.episode_length)
        logger_dict['Train/base_height'] = self.mean_base_height.mean().item()
        logger_dict['Train/AdaSmpl'] = self.p_smpl

        if self.cfg.logger_backend == 'wandb':
            wandb.log(logger_dict, step=self.cur_it)
        elif self.cfg.logger_backend == 'tensorboard':
            for t, v in logger_dict.items():
                self.logger.add_scalar(t, v, global_step=self.cur_it)
            self.logger.flush()

        # logging string to print
        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + self.cfg.max_iterations} \033[0m "
        fps = int(self.num_steps_per_env * self.cfg.env.num_envs / iteration_time)
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (self.cfg.max_iterations - curr_it)
        log_string = (
            f"""{'*' * width}\n"""
            f"""{progress.center(width, ' ')}\n\n"""
            f"""{'Experiment:':>{pad}} {self.exptid}\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_steps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s {self.collection_time:.2f}s {self.learn_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta // 60:.0f} mins {eta % 60:.1f} s\n"""
            f"""{'CUDA allocated:':>{pad}} {torch.cuda.memory_allocated() / 1024 / 1024:.2f}\n"""
            f"""{'CUDA reserved:':>{pad}} {torch.cuda.memory_reserved() / 1024 / 1024:.2f}\n"""
        )
        print(log_string)

    def play_act(self, obs, **kwargs):
        self.algorithm.actor.eval()
        return self.algorithm.play_act(obs, **kwargs)

    def save(self, path, infos=None):
        state_dict = self.algorithm.save()
        state_dict['iter'] = self.cur_it
        state_dict['infos'] = infos
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from", path)

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.start_it = loaded_dict['iter']
        infos = self.algorithm.load(loaded_dict, load_optimizer)

        print("*" * 80)
        return infos
