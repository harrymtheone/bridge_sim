from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Callable, Sequence

import torch
from isaaclab.managers import ManagerBase, ManagerTermBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class MotionTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING

    params: dict = {}


class MotionGenerator(ManagerBase):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        self._term_names: list[str] = []
        self._term_cfgs: list[MotionTermCfg] = []

        super().__init__(cfg, env)

        self._motion_buffer: dict[str, torch.Tensor] = {}

    def __str__(self):  # TODO: str is not finished
        return ""

    def active_terms(self) -> list[str]:
        return self._term_names

    def get_motion(self, term_name: str | None = None) -> torch.Tensor:
        if term_name is None:
            return next(iter(self._motion_buffer.values()))

        return self._motion_buffer[term_name]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self.compute()
        return {}

    def compute(self) -> dict[str, torch.Tensor]:
        self._motion_buffer: dict[str, torch.Tensor] = {}

        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            self._motion_buffer[term_name] = term_cfg.func(self._env, **term_cfg.params)

        return self._motion_buffer

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg = self.cfg
        else:
            cfg = self.cfg.__dict__

        # iterate over all the terms
        for term_name, term_cfg in cfg.items():
            # check for non config
            if term_cfg is None:
                raise TypeError(f"Motion term for {term_name} is None!")

            # check for valid config type
            if not isinstance(term_cfg, MotionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type MotionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg)

            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
