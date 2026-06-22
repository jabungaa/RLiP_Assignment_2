"""PyTorch PPO agent for the continuous grid-world environment.

The agent uses separate actor and critic multilayer perceptrons:

    state features -> actor MLP  -> action logits
    state features -> critic MLP -> V(s)

It keeps the public `BaseAgent` interface used by the project while doing PPO
updates with PyTorch autograd.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents import BaseAgent


NUM_ACTIONS = 4  # 0=forward, 1=turn left, 2=turn right, 3=backward

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu":  nn.ELU,
    "gelu": nn.GELU,
}
# Orthogonal init gain per activation (used for hidden layers)
_ACT_GAINS: dict[str, float] = {
    "tanh": 5 / 3,
    "relu": np.sqrt(2.0),
    "elu":  np.sqrt(2.0),
    "gelu": np.sqrt(2.0),
}


def _make_activation(name: str) -> nn.Module:
    cls = _ACTIVATIONS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown activation {name!r}. Choose from {list(_ACTIVATIONS)}")
    return cls()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        output_dim: int,
        activation: str = "tanh",
    ):
        super().__init__()
        gain = _ACT_GAINS.get(activation.lower(), np.sqrt(2.0))
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_make_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        if isinstance(self.net[-1], nn.Linear):
            nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPO_agent(BaseAgent):
    def __init__(
        self,
        grid: str | Path | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        rollout_steps: int = 128,
        hidden_sizes: tuple[int, ...] | list[int] = (64, 128),
        advantage_norm: bool = True,
        reward_scale: float = 1.0,
        max_grad_norm: float | None = 0.5,
        activation: str = "tanh",
        fourier_freqs: int = 0,
        state_size: int = 22,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ):
        """Create a PyTorch PPO actor-critic agent.

        Args:
            grid: Unused compatibility argument for old runners.
            gamma: Discount factor.
            gae_lambda: Lambda for generalized advantage estimation.
            clip_epsilon: PPO policy-ratio clipping range.
            policy_lr: Actor Adam learning rate.
            value_lr: Critic Adam learning rate.
            entropy_coef: Entropy bonus weight.
            value_coef: Value loss multiplier.
            update_epochs: Optimization passes over each rollout.
            minibatch_size: Number of transitions per gradient step. The
                rollout is split into chunks of this size each epoch.
            rollout_steps: Transitions collected (across episodes) before a
                PPO update. Episode boundaries inside the buffer are handled
                with GAE chain cuts and time-limit bootstrapping.
            hidden_sizes: Hidden layer widths for actor and critic MLPs.
            advantage_norm: Normalize advantages inside each rollout.
            reward_scale: All rewards are divided by this before storage, so
                the critic sees targets of magnitude ~1. Set it to the goal
                reward of the active reward function, e.g. 10 for `default`
                or 10000 for `high`.
            max_grad_norm: Optional gradient norm clipping.
            state_size: Continuous environment state size. The environment
                already normalizes the state, so this is just the raw count
                of features: 4 base (x, y, cos θ, sin θ) + N_RAYS lidar = 22.
            seed: Optional RNG seed for NumPy and PyTorch.
            device: Optional torch device, e.g. "cpu" or "cuda"."""
        super().__init__()

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rollout_steps = rollout_steps
        self.hidden_sizes = tuple(hidden_sizes)
        self.advantage_norm = advantage_norm
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.reward_scale = float(reward_scale)
        self.max_grad_norm = max_grad_norm
        self.activation = activation
        self.fourier_freqs = fourier_freqs
        if state_size < 4:
            raise ValueError("state_size must include at least x, y, cos θ, sin θ")
        self.state_size = int(state_size)
        self.lidar_dim = self.state_size - 4
        self.device = torch.device(device or "cpu")

        # state is already normalized; optionally append Fourier features for x, y
        self.input_dim = self.state_size + 4 * self.fourier_freqs
        self.actor = MLP(self.input_dim, self.hidden_sizes, NUM_ACTIONS, activation=activation).to(self.device)
        self.critic = MLP(self.input_dim, self.hidden_sizes, 1, activation=activation).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=value_lr)

        self.train_mode = True

        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.last_log_prob: float | None = None
        self.last_value: float | None = None

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        # dones[i] is True when the GAE chain is cut after step i (true
        # terminal OR episode truncation). next_values[i] is the bootstrap
        # value used at that cut: 0.0 for a true terminal, V(s_next) for a
        # time-limit truncation.
        self.dones: list[bool] = []
        self.next_values: list[float] = []
        self.old_log_probs: list[float] = []
        self.old_values: list[float] = []

    def _encode_state(self, state) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim != 1 or state_arr.shape[0] != self.state_size:
            raise ValueError(
                f"Expected state shape ({self.state_size},), got {state_arr.shape}"
            )

        if self.fourier_freqs == 0:
            return state_arr

        # state[0]=x_norm, state[1]=y_norm — already in [0, 1]
        x_norm, y_norm = float(state_arr[0]), float(state_arr[1])
        fourier: list[float] = []
        for k in range(self.fourier_freqs):
            f = math.pi * (2 ** k)
            fourier += [
                math.sin(f * x_norm), math.cos(f * x_norm),
                math.sin(f * y_norm), math.cos(f * y_norm),
            ]
        return np.concatenate([state_arr, np.array(fourier, dtype=np.float32)])

    def _state_tensor(self, states) -> torch.Tensor:
        state_arr = np.asarray(states, dtype=np.float32)
        if state_arr.ndim == 1:
            encoded = self._encode_state(state_arr)[None, :]
        else:
            encoded = np.stack([self._encode_state(state) for state in state_arr])
        return torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

    def _actor_logits(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.actor(state_tensor)

    def _critic_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.critic(state_tensor)

    def _distribution(self, states) -> Categorical:
        state_tensor = self._state_tensor(states)
        logits = self._actor_logits(state_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "PPO actor produced non-finite logits: training has diverged. "
                "Lower policy_lr and/or check reward scaling."
            )
        return Categorical(logits=logits)

    def _value(self, state) -> float:
        with torch.no_grad():
            value = self._critic_values(self._state_tensor(state)).squeeze(-1)
        return float(value.item())

    def new_episode(self, state=None):
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

    def take_action(self, state) -> int:
        with torch.no_grad():
            dist = self._distribution(state)
            if self.train_mode:
                action_tensor = dist.sample()
            else:
                action_tensor = torch.argmax(dist.probs, dim=-1)

            log_prob = dist.log_prob(action_tensor)
            value = self._critic_values(self._state_tensor(state)).squeeze(-1)

        action = int(action_tensor.item())
        self.last_state = state
        self.last_action = action
        self.last_log_prob = float(log_prob.item())
        self.last_value = float(value.item())
        return action

    def update(self, state, reward: float, action: int, done: bool = False):
        if self.last_state is None or self.last_action is None:
            return

        scaled_reward = float(reward) / self.reward_scale
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        self.rewards.append(scaled_reward)
        self.dones.append(done)
        self.next_values.append(0.0)  # true terminals bootstrap with 0
        self.old_log_probs.append(float(self.last_log_prob))
        self.old_values.append(float(self.last_value))

        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

        # Episode ends no longer flush the buffer; transitions accumulate
        # across episodes until rollout_steps is reached.
        if len(self.states) >= self.rollout_steps:
            if not done:
                # Buffer full mid-episode: bootstrap from the state we just
                # landed in (time-limit style cut).
                self._mark_boundary(state)
            self._flush()

    def _mark_boundary(self, bootstrap_state):
        """Cut the GAE chain after the last stored transition, bootstrapping
        with V(bootstrap_state). No-op if the chain is already cut."""
        if self.dones and not self.dones[-1]:
            self.dones[-1] = True
            self.next_values[-1] = self._value(bootstrap_state)

    def finish_rollout(self, bootstrap_state=None):
        """Mark an episode boundary and flush the buffer when appropriate.

        With `bootstrap_state` (episode truncated by a step limit): mark the
        boundary; optimize only once the buffer holds rollout_steps
        transitions. With no argument: force a final flush (end of training).
        """
        if not self.states:
            return

        if bootstrap_state is not None:
            self._mark_boundary(bootstrap_state)
            if len(self.states) >= self.rollout_steps:
                self._flush()
            return

        self._flush()

    def _flush(self):
        if not self.states:
            return

        states = list(self.states)
        actions = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=bool)
        boundary_values = np.array(self.next_values, dtype=np.float32)
        old_log_probs = torch.as_tensor(
            self.old_log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        old_values = np.array(self.old_values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0  # tail is normally a marked boundary; 0 is the safe fallback
        for idx in range(len(rewards) - 1, -1, -1):
            if dones[idx]:
                # Chain cut: terminal (boundary value 0) or truncation
                # (boundary value V(s_next)).
                delta = rewards[idx] + self.gamma * boundary_values[idx] - old_values[idx]
                last_gae = delta
            else:
                delta = rewards[idx] + self.gamma * next_value - old_values[idx]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[idx] = last_gae
            next_value = old_values[idx]

        returns = advantages + old_values
        if self.advantage_norm and len(advantages) > 1:
            std = float(np.std(advantages))
            if std > 1e-8:
                advantages = (advantages - float(np.mean(advantages))) / (std + 1e-8)

        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self._optimize(states, actions, old_log_probs, adv_tensor, ret_tensor)

        self._clear_rollout()

    def _optimize(
        self,
        states: list[np.ndarray],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        n = len(states)
        mb = min(self.minibatch_size, n)

        for _ in range(self.update_epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, mb):
                idx = indices[start: start + mb]
                mb_states = [states[int(i.item())] for i in idx]

                dist = self._distribution(mb_states)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                values = self._critic_values(self._state_tensor(mb_states)).squeeze(-1)

                ratios = torch.exp(new_log_probs - old_log_probs[idx])
                unclipped = ratios * advantages[idx]
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages[idx]

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(values, returns[idx])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def _clear_rollout(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_values.clear()
        self.old_log_probs.clear()
        self.old_values.clear()

    def set_training(self, enabled: bool):
        self.train_mode = enabled
        self.actor.train(enabled)
        self.critic.train(enabled)
