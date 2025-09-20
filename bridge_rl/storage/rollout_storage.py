from __future__ import annotations

import torch


class DataBuf:
    def __init__(self, buf_length, num_envs, shape, dtype, device):
        self.buf_length = buf_length
        self.num_envs = num_envs
        self.device = device

        self.buf = torch.zeros(buf_length, num_envs, *shape, dtype=dtype, device=self.device)

    def set(self, idx, value):
        self.buf[idx] = value

    def get(self, slice_):
        return self.buf[slice_]

    def flatten_get(self, idx):
        return self.buf.flatten(0, 1)[idx]


class HiddenBuf:
    def __init__(self, buf_length, num_envs, num_layers, hidden_size, dtype, device):
        self.buf_length = buf_length
        self.num_envs = num_envs
        self.device = device

        self.buf = torch.zeros(buf_length, num_layers, num_envs, hidden_size, dtype=dtype, device=self.device)

    def set(self, idx, value):
        if value is None:
            self.buf[idx] = 0.
        else:
            self.buf[idx] = value

    def get(self, slice_):
        return self.buf[0][slice_].contiguous()


class ObsTransBuf:
    def __init__(self,
                 obs_shape: dict[str, int | tuple[int, ...]],
                 buf_length,
                 num_envs,
                 dtype: torch.dtype,
                 device: torch.device):
        self.buf_length = buf_length
        self.num_envs = num_envs
        self.device = device

        self.storage = {}
        for name, dims in obs_shape.items():
            if isinstance(dims, int):
                dims = (dims,)

            self.storage[name] = torch.zeros(self.buf_length, self.num_envs, *dims, dtype=dtype, device=self.device)

    def set(self, idx, obs):
        for n, v in obs.items():
            if n not in self.storage:
                self.storage[n] = torch.zeros(self.buf_length, *v.shape, dtype=v.dtype, device=self.device)

            self.storage[n][idx] = v

    def flatten_get(self, idx):
        raise NotImplementedError
        param = [v.flatten(0, 1)[idx] for v in self.storage.values()]
        return self.obs_class(*param)

    def get(self, slice_):
        return {k: v[slice_] for k, v in self.storage.items()}


class RolloutStorage:
    def __init__(self,
                 obs_shape: dict[str, tuple[int, ...]],
                 num_actions: int,
                 storage_length,
                 num_envs,
                 device,
                 dtype=torch.float):
        self.storage_length = storage_length
        self.num_envs = num_envs
        self.dtype = dtype
        self.device = device
        self._step = 0

        self.storage = {
            'observations': ObsTransBuf(obs_shape, self.storage_length, self.num_envs, dtype, self.device),
            'rewards': DataBuf(self.storage_length, self.num_envs, (1,), dtype, self.device),
            'dones': DataBuf(self.storage_length, self.num_envs, (1,), dtype, self.device),
            'values': DataBuf(self.storage_length, self.num_envs, (1,), dtype, self.device),
            'returns': DataBuf(self.storage_length, self.num_envs, (1,), dtype, self.device),
            'advantages': DataBuf(self.storage_length, self.num_envs, (1,), dtype, self.device),
            'actions': DataBuf(self.storage_length, self.num_envs, (num_actions,), dtype, self.device),
            'actions_log_prob': DataBuf(self.storage_length, self.num_envs, (num_actions,), dtype, self.device),
            'action_mean': DataBuf(self.storage_length, self.num_envs, (num_actions,), dtype, self.device),
            'action_sigma': DataBuf(self.storage_length, self.num_envs, (num_actions,), dtype, self.device),
        }

    def register_data_buffer(self, name, data_shape: int | tuple[int, ...], dtype: torch.dtype):
        self.storage[name] = DataBuf(self.storage_length, self.num_envs, data_shape, dtype, self.device)

    def register_hidden_state_buffer(self, name, num_layers: int, hidden_size: int):
        self.storage[name] = HiddenBuf(self.storage_length, self.num_envs, num_layers, hidden_size, self.dtype, self.device)

    def add_transitions(self, name, transition: torch.Tensor | dict[str, torch.Tensor]):
        if self._step >= self.storage_length:
            raise AssertionError("Rollout buffer overflow")

        self.storage[name].set(self._step, transition)

    def flush(self):
        self._step += 1

    def clear(self):
        self._step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        dones_buf = self.storage['dones'].buf.float()
        rewards_buf = self.storage['rewards'].buf
        values_buf = self.storage['values'].buf
        returns_buf = self.storage['returns'].buf
        advantages_buf = self.storage['advantages'].buf

        for step in reversed(range(self.storage_length)):
            if step == self.storage_length - 1:
                next_values = last_values
            else:
                next_values = self.storage['values'].get(step + 1)

            next_is_not_terminal = 1.0 - dones_buf[step]
            delta = rewards_buf[step] + next_is_not_terminal * gamma * next_values - values_buf[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns_buf[step] = advantage + values_buf[step]

        # Compute and normalize the advantages
        advantages_buf[:] = returns_buf - values_buf
        advantages_buf[:] = (advantages_buf - advantages_buf.mean()) / (advantages_buf.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.step < self.num_rollout_steps - 1:
            raise AssertionError('why the buffer is not full?')

        batch_size = self.num_envs * self.num_rollout_steps
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(batch_size)

        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                batch_dict = {n: v.flatten_get(batch_idx) for n, v in self.storage.items()}
                batch_dict['returns'] = returns[batch_idx]
                batch_dict['advantages'] = advantages[batch_idx]

                yield batch_dict

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        valid_mask = ~(torch.cumsum(self.storage['dones'].buf, dim=0) > 0)

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                batch_dict = {'masks': valid_mask[:, start: stop]}

                for n, v in self.storage.items():
                    batch_dict[n] = v.get((slice(None), slice(start, stop)))

                yield batch_dict
