# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers import ManagerBase, ActionTerm, ActionTermCfg
from prettytable import PrettyTable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ActionManagerV2(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # create buffers to store actions
        self._action: dict[str, torch.Tensor] = {}
        self._prev_action: dict[str, torch.Tensor] = {}

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> dict[str, torch.Tensor]:
        """The actions sent to the environment as a dictionary mapping term names to tensors."""
        return self._action

    @property
    def prev_action(self) -> dict[str, torch.Tensor]:
        """The previous actions sent to the environment as a dictionary mapping term names to tensors."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    @property
    def get_IO_descriptors(self) -> list[dict[str, Any]]:
        """Get the IO descriptors for the action manager.

        Returns:
            A dictionary with keys as the term names and values as the IO descriptors.
        """

        data = []

        for term_name, term in self._terms.items():
            try:
                data.append(term.IO_descriptor.__dict__.copy())
            except Exception as e:
                print(f"Error getting IO descriptor for term '{term_name}': {e}")

        formatted_data = []
        for item in data:
            name = item.pop("name")
            formatted_item = {"name": name, "extras": item.pop("extras")}
            print(item["export"])
            if not item.pop("export"):
                continue
            for k, v in item.items():
                # Check if v is a tuple and convert to list
                if isinstance(v, tuple):
                    v = list(v)
                if k in ["description", "units"]:
                    formatted_item["extras"][k] = v
                else:
                    formatted_item[k] = v
            formatted_data.append(formatted_item)

        return formatted_data

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        for term_name in self._action:
            self._prev_action[term_name][env_ids] = 0.0
            self._action[term_name][env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: dict[str, torch.Tensor]):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
                - dict[str, torch.Tensor]: Dictionary mapping term names to their actions
        """
        # validate that all required terms are present
        missing_terms = set(self._term_names) - set(action.keys())
        if missing_terms:
            raise ValueError(f"Missing action terms: {missing_terms}. Expected terms: {self._term_names}")

        extra_terms = set(action.keys()) - set(self._term_names)
        if extra_terms:
            raise ValueError(f"Unknown action terms: {extra_terms}. Expected terms: {self._term_names}")

        # store previous actions
        for term_name in self._action:
            self._prev_action[term_name][:] = self._action[term_name]

        # validate and store new actions
        for term_name, term in self._terms.items():
            term_action = action[term_name].to(self.device)

            # validate action shape
            expected_shape = (self.num_envs, term.action_dim)
            if term_action.shape != expected_shape:
                raise ValueError(
                    f"Invalid action shape for term '{term_name}'. "
                    f"Expected: {expected_shape}, received: {term_action.shape}"
                )

            # store in dict format
            self._action[term_name][:] = term_action

            # process the action
            term.process_actions(term_action)

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term

            # initialize action tensors for this term
            self._action[term_name] = torch.zeros((self.num_envs, term.action_dim), device=self.device)
            self._prev_action[term_name] = torch.zeros_like(self._action[term_name])
