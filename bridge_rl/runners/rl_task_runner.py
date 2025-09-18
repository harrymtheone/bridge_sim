from __future__ import annotations

import os
import time
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from torch.utils.tensorboard import SummaryWriter

from bridge_env.envs import BridgeEnv
from bridge_rl.utils import EpisodeLogger

if TYPE_CHECKING:
    from . import RLTaskCfg


class RLRunner:
    def __init__(self, cfg: RLTaskCfg, args: Namespace):
        cfg.validate()
        self.cfg = cfg

        self.env = BridgeEnv(cfg=cfg.env_cfg)
        self.device = args.device

        # Create algorithm
        self.algorithm = cfg.algorithm_cfg.class_type(cfg.algorithm_cfg, env=self.env)

        # Timing tracking
        self.collection_time = -1
        self.learn_time = -1
        self.tot_time = 0
        self.tot_steps = 0

        self.start_it = 0
        self.cur_it = 0

        # Initialize episode logger
        self.episode_logger = EpisodeLogger(num_envs=self.env.num_envs)
        self._prepare_log_dir(args)

    def learn(self):
        self.algorithm.train()

        observations, infos = self.env.reset()
        self.episode_logger.reset()

        for self.cur_it in range(self.start_it, self.start_it + self.cfg.max_iterations):
            start_time = time.time()

            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_update):
                    observations['use_estimated_values'] = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)  # TODO: not finished here?!

                    actions = self.algorithm.act(observations)
                    observations, rewards, terminated, timeouts, infos = self.env.step(actions)

                    if self.cur_it < 100:
                        rewards_clipped = rewards.clip(min=0.)
                    else:
                        rewards_clipped = rewards

                    self.algorithm.process_env_step(rewards_clipped, terminated, timeouts, infos)
                    self.episode_logger.step(rewards, terminated, timeouts)

                self.algorithm.compute_returns(observations)

            self.collection_time = time.time() - start_time
            start_time = time.time()

            update_infos = self.algorithm.update()

            self.learn_time = time.time() - start_time

            self.log(infos, update_infos)

            if self.cur_it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.model_dir, f'model_{self.cur_it}.pt'))
            self.save(os.path.join(self.model_dir, 'latest.pt'))

    def _prepare_log_dir(self, args):
        algorithm_name = self.cfg.algorithm_cfg.class_type.__name__
        alg_dir: str = os.path.join(args.log_root, args.proj_name, algorithm_name)  # noqa
        self.model_dir = os.path.join(alg_dir, args.exptid)
        os.makedirs(self.model_dir, exist_ok=True)

        if args.resume or args.resumeid:
            if args.resumeid is None:
                args.resumeid = args.exptid

            resume_dir = os.path.join(alg_dir, args.resumeid)

            if not os.path.isdir(resume_dir):
                raise ValueError(f"resume directory \"{resume_dir}\" is not a directory")

            if args.checkpoint is None:
                models = [file for file in os.listdir(resume_dir) if file.startswith("model")]

                if 'latest.pt' in os.listdir(resume_dir):
                    model_name = 'latest.pt'
                elif len(models) > 0:
                    models.sort(key=lambda m: '{0:0>15}'.format(m))
                    model_name = models[-1]
                else:
                    raise ValueError(f"No checkpoint found at \"{resume_dir}\"")

            else:
                model_name = "model_{}.pt".format(args.checkpoint)

            resume_path = os.path.join(resume_dir, model_name)

            self.load(resume_path)

        if self.cfg.logger_backend == 'tensorboard':
            tensorboard_file = list(Path(self.model_dir).glob('events.*'))
            if tensorboard_file:
                raise FileExistsError("Logging directory not empty!")

            self.logger = SummaryWriter(log_dir=self.model_dir)

    def save(self, path: str, infos: dict = None):
        state_dict = self.algorithm.save()
        state_dict['iter'] = self.cur_it
        state_dict['infos'] = infos
        torch.save(state_dict, path)

    def log(self, infos, update_infos, width=80, pad=35):
        # Update total step count and time
        self.tot_steps += self.cfg.num_steps_per_update * self.env.num_envs
        iteration_time = self.collection_time + self.learn_time
        self.tot_time += iteration_time

        # Build logger dict
        logger_dict = infos['log']
        logger_dict.update(update_infos)

        logger_dict.update(self.episode_logger.get_logging_dict())

        if self.cfg.logger_backend == 'wandb':
            import wandb
            wandb.log(logger_dict, step=self.cur_it)

        elif self.cfg.logger_backend == 'tensorboard':
            for t, v in logger_dict.items():
                self.logger.add_scalar(t, v, global_step=self.cur_it)
            self.logger.flush()

        # Print progress information
        progress = f" \033[1m Learning iteration {self.cur_it}/{self.start_it + self.cfg.max_iterations} \033[0m "
        fps = int(self.cfg.num_steps_per_update * self.env.num_envs / iteration_time)
        curr_it = self.cur_it - self.start_it
        eta = self.tot_time / (curr_it + 1) * (self.cfg.max_iterations - curr_it)

        print(
            f"""{'*' * width}\n"""
            f"""{progress.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_steps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s {self.collection_time:.2f}s {self.learn_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {eta // 60:.0f} mins {eta % 60:.1f} s\n"""
            f"""{'CUDA allocated:':>{pad}} {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB\n"""
            f"""{'CUDA reserved:':>{pad}} {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB\n"""
        )

    def play(self):
        self.algorithm.eval()

        observations, infos = self.env.reset()
        with torch.inference_mode():
            while True:
                observations['use_estimated_values'] = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)  # TODO: not finished here?!

                rtn = self.algorithm.play_act(observations)

                # actions = rtn['actions']

                actions = self.env.motion_generator.get_motion('ref_motion') - self.env.scene['robot'].data.default_joint_pos

                observations, rewards, terminated, timeouts, infos = self.env.step(actions)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from", path)

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.start_it = loaded_dict['iter']
        infos = self.algorithm.load(loaded_dict, load_optimizer)

        print("*" * 80)
        return infos
