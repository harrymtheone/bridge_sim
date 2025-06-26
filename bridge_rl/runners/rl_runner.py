from __future__ import annotations

import os
from argparse import Namespace
from typing import TYPE_CHECKING

import torch
import wandb
from isaaclab.envs import ManagerBasedRLEnv
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from bridge_rl.runners.rl_runner import RLRunnerCfg


class RLRunner:
    def __init__(self, cfg: RLRunnerCfg, env: ManagerBasedRLEnv, args: Namespace):
        cfg.validate()

        self.cfg = cfg
        self.env = env
        self.device = args.device

        self._prepare_log_dir(args)

        # Create algorithm
        self.algorithm = cfg.algorithm_cfg.class_type(self.cfg.algorithm_cfg, env=env)

        self.start_it = 0
        self.cur_it = 0

    def learn(self):
        self.algorithm.train()

        observations, infos = self.env.reset()

        for self.cur_it in range(self.start_it, self.start_it + self.cfg.max_iterations):

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg.num_steps_per_env):
                    actions = self.algorithm.act(observations)
                    observations, rewards, terminated, timeouts, infos = self.env.step(actions)
                    self.algorithm.process_env_step(rewards, terminated, timeouts, infos)

                # Learning step
                self.algorithm.compute_returns(observations)

            update_infos = self.algorithm.update()
            self.log(infos, update_infos)

            if self.cur_it % self.cfg.save_interval == 0:
                self.save(os.path.join(self.model_dir, f'model_{self.cur_it}.pt'))
            self.save(os.path.join(self.model_dir, 'latest.pt'))

    def _prepare_log_dir(self, args):
        alg_dir = os.path.join(args.log_root, args.proj_name, self.cfg.algorithm_cfg.class_type.__name__)
        self.model_dir = os.path.join(alg_dir, args.exptid)
        os.makedirs(self.model_dir, exist_ok=True)

        if args.resumeid:
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
            self.logger = SummaryWriter(log_dir=self.model_dir)

    def save(self, path: str, infos: dict = None):
        state_dict = self.algorithm.save()
        state_dict['iter'] = self.cur_it
        state_dict['infos'] = infos
        torch.save(state_dict, path)

    def log(self, infos, update_infos):
        logger_dict = infos['log']
        logger_dict.update(update_infos)

        if self.cfg.logger_backend == 'wandb':
            wandb.log(logger_dict, step=self.cur_it)
        elif self.cfg.logger_backend == 'tensorboard':
            for t, v in logger_dict.items():
                self.logger.add_scalar(t, v, global_step=self.cur_it)
            self.logger.flush()

    def play_act(self, obs, **kwargs):
        self.algorithm.actor.eval()
        return self.algorithm.play_act(obs, **kwargs)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from", path)

        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.start_it = loaded_dict['iter']
        infos = self.algorithm.load(loaded_dict, load_optimizer)

        print("*" * 80)
        return infos
