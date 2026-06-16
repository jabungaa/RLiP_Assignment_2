"""PyTorch PPO agent for the grid-world environment.

The agent uses separate actor and critic multilayer perceptrons:

    state features -> actor MLP  -> action logits
    state features -> critic MLP -> V(s)

It keeps the public `BaseAgent` interface used by the project while doing PPO
updates with PyTorch autograd.
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from agents import BaseAgent

NUM_ACTIONS = 3

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


class SharedActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        num_actions: int,
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

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.actor_head = nn.Linear(prev_dim, num_actions)
        self.critic_head = nn.Linear(prev_dim, 1)

        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.actor_head(features), self.critic_head(features)


class PPOReplayBuffer:
    """Circular buffer of pre-computed (state, action, log_prob, advantage, return) tuples.

    Advantages and log_probs are stored from the time of collection.  The PPO
    clipped IS ratio exp(log π_new − log π_old) corrects for the policy drift
    when the buffer is sampled in later updates.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self._pos = 0
        self._size = 0
        self._states: list = [None] * capacity
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._log_probs = np.zeros(capacity, dtype=np.float32)
        self._advantages = np.zeros(capacity, dtype=np.float32)
        self._returns = np.zeros(capacity, dtype=np.float32)

    def push_batch(
        self,
        states: list,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        adv_np = advantages.cpu().numpy()
        ret_np = returns.cpu().numpy()
        for i in range(len(states)):
            self._states[self._pos] = states[i]
            self._actions[self._pos] = actions_np[i]
            self._log_probs[self._pos] = log_probs_np[i]
            self._advantages[self._pos] = adv_np[i]
            self._returns[self._pos] = ret_np[i]
            self._pos = (self._pos + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int, device: torch.device):
        idx = self.rng.integers(self._size, size=n)
        states = [self._states[i] for i in idx]
        actions = torch.as_tensor(self._actions[idx], dtype=torch.long, device=device)
        log_probs = torch.as_tensor(self._log_probs[idx], dtype=torch.float32, device=device)
        advantages = torch.as_tensor(self._advantages[idx], dtype=torch.float32, device=device)
        returns = torch.as_tensor(self._returns[idx], dtype=torch.float32, device=device)
        return states, actions, log_probs, advantages, returns

    def __len__(self) -> int:
        return self._size


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
        rollout_steps: int = 128,
        minibatch_size: int | None = None,
        hidden_sizes: tuple[int, ...] | list[int] = (64, 128),
        advantage_norm: bool = True,
        reward_scale: float = 1.0,
        reward_clip: float | None = 10.0,
        max_grad_norm: float | None = 0.5,
        replay_buffer_size: int | None = None,
        activation: str = "tanh",
        fourier_freqs: int = 0,
        fourier_normalized: bool = True,
        seed: int | None = None,
        device: str | torch.device | None = None,
        network_mode: str = "separate",
    ):
        """Create a PyTorch PPO actor-critic agent.

        Args:
            grid: Optional `.npy` grid file. Enables invalid-action masking.
                The policy observation does not include target position.
            gamma: Discount factor.
            gae_lambda: Lambda for generalized advantage estimation.
            clip_epsilon: PPO policy-ratio clipping range.
            policy_lr: Actor Adam learning rate.
            value_lr: Critic Adam learning rate.
            entropy_coef: Entropy bonus weight.
            value_coef: Value loss multiplier.
            update_epochs: Optimization passes over each rollout.
            rollout_steps: Transitions collected (across episodes) before a
                PPO update. Episode boundaries inside the buffer are handled
                with GAE chain cuts and time-limit bootstrapping.
            minibatch_size: Optional mini-batch size. Defaults to full batch.
            hidden_sizes: Hidden layer widths for actor and critic MLPs.
            advantage_norm: Normalize advantages inside each rollout.
            reward_scale: All rewards are divided by this before clipping and
                storage, so the critic sees targets of magnitude ~1. Set it to
                the goal reward of the active reward function (e.g. 1e8 for
                `zero_penalty_reward`, 1e4 for `low_penalty_reward`).
            reward_clip: Optional reward clipping, applied after scaling.
            max_grad_norm: Optional gradient norm clipping.
            seed: Optional RNG seed for NumPy and PyTorch.
            device: Optional torch device, e.g. "cpu" or "cuda".
            network_mode: "separate" uses independent actor/critic MLPs.
                "shared" uses one shared backbone with actor and critic heads.
        """
        super().__init__()

        if network_mode not in ("separate", "shared"):
            raise ValueError("network_mode must be either 'separate' or 'shared'")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.advantage_norm = advantage_norm
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.reward_scale = float(reward_scale)
        self.reward_clip = reward_clip
        self.max_grad_norm = max_grad_norm
        self.activation = activation
        self.fourier_freqs = fourier_freqs
        self.fourier_normalized = fourier_normalized
        self.network_mode = network_mode
        self.device = torch.device(device or "cpu")
        self.rng = np.random.default_rng(seed)
        self.replay_buffer: PPOReplayBuffer | None = (
            PPOReplayBuffer(replay_buffer_size, self.rng)
            if replay_buffer_size is not None else None
        )

        self.grid = None
        self.rows = None
        self.cols = None
        self.target_positions: list[tuple[float, float, float, float]] = []
        if grid is not None:
            self.grid = np.load(Path(grid))
            self.rows, self.cols = self.grid.shape
            self.target_positions = [
                (float(c), float(r), float(c) + 1.0, float(r) + 1.0)
                for c, r in np.argwhere(self.grid == 3)
            ]

        if fourier_freqs == 0:
            self.input_dim = 21                         # raw (row_norm, col_norm)
        elif fourier_normalized:
            self.input_dim = 21 + 4 * fourier_freqs     # raw coords + sin/cos per freq
        else:
            self.input_dim = 19 + 4 * fourier_freqs          # sin/cos only (no raw integers)
        if self.network_mode == "shared":
            self.model = SharedActorCritic(
                self.input_dim,
                self.hidden_sizes,
                NUM_ACTIONS,
                activation=activation,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=policy_lr)
            self.actor = self.model
            self.critic = self.model
        else:
            self.actor = MLP(self.input_dim, self.hidden_sizes, NUM_ACTIONS, activation=activation).to(self.device)
            self.critic = MLP(self.input_dim, self.hidden_sizes, 1, activation=activation).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=value_lr)

        # Evaluation helpers in this project set `epsilon = 0.0` when present.
        # For PPO this means "act greedily".
        self.epsilon = 1.0
        self.train_mode = True

        self.last_state: tuple[float, ...] | None = None
        self.last_action: int | None = None
        self.last_log_prob: float | None = None
        self.last_value: float | None = None

        self.states: list[tuple[float, ...]] = []
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

    def _encode_state(self, state: tuple[float, ...]) -> np.ndarray:
        row, col = float(state[0]), float(state[1])
        row_denom = max(1, self.rows - 1) if self.rows else max(1.0, row)
        col_denom = max(1, self.cols - 1) if self.cols else max(1.0, col)
        row_norm = row / row_denom
        col_norm = col / col_denom
        features = list(state)
        if self.fourier_freqs == 0:
            return np.array(features, dtype=np.float32)

        if self.fourier_normalized:
            # NeRF-style: frequencies π, 2π, 4π, ..., 2^(K-1)·π on [0,1] coords.
            # Adjacent cells (step ≈ 1/(N-1)) differ by 2^k·π/(N-1) radians.
            # At k=3 on a 20-cell axis that's 8π/19 ≈ 1.3 rad — easily distinguishable.
            r, c = row_norm, col_norm
            #features: list[float] = [r, c]
            for k in range(self.fourier_freqs):
                f = (2 ** k) * math.pi
                features += [math.sin(f * r), math.cos(f * r),
                             math.sin(f * c), math.cos(f * c)]
        else:
            # Raw-integer style: base frequency π/2 so adjacent integer steps
            # (Δ=1) shift by π/2 — a quarter period, maximally distinct.
            # Lower frequencies (π/4, π/8, …) add coarser spatial context.
            r, c = row, col
            features = features[2:]  # theta + lidar rays
            for k in range(self.fourier_freqs):
                f = math.pi / (2 ** k)   # π, π/2, π/4, π/8, …
                features += [math.sin(f * r), math.cos(f * r),
                             math.sin(f * c), math.cos(f * c)]

        return np.array(features, dtype=np.float32)

    def _state_tensor(self, states: list[tuple[float, ...]] | tuple[float, ...]) -> torch.Tensor:
        if isinstance(states, tuple):
            encoded = self._encode_state(states)[None, :]
        else:
            encoded = np.stack([self._encode_state(state) for state in states])
        return torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

    def _actor_logits(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if self.network_mode == "shared":
            logits, _values = self.model(state_tensor)
            return logits
        return self.actor(state_tensor)

    def _critic_values(self, state_tensor: torch.Tensor) -> torch.Tensor:
        if self.network_mode == "shared":
            _logits, values = self.model(state_tensor)
            return values
        return self.critic(state_tensor)

    def _distribution(self, states: list[tuple[float, ...]] | tuple[float, ...]) -> Categorical:
        state_tensor = self._state_tensor(states)
        logits = self._actor_logits(state_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "PPO actor produced non-finite logits: training has diverged. "
                "Lower policy_lr and/or check reward scaling."
            )
        return Categorical(logits=logits)

    def _value(self, state: tuple[float, ...]) -> float:
        with torch.no_grad():
            value = self._critic_values(self._state_tensor(state)).squeeze(-1)
        return float(value.item())

    def _is_terminal_transition(
        self,
        next_state: tuple[float, ...],
        reward: float,
    ) -> bool:
        if self.target_positions:
            x, y = float(next_state[0]), float(next_state[1])
            for x_min, y_min, x_max, y_max in self.target_positions:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True

        # Existing reward functions only use positive rewards for target cells.
        return reward > 0

    def new_episode(self, state: tuple[float, ...] | None = None):
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

    def take_action(self, state: tuple[float, ...]) -> int:
        with torch.no_grad():
            state = tuple(state)
            dist = self._distribution(state)
            if self.train_mode and self.epsilon != 0.0:
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

    def update(self, state: tuple[float, ...], reward: float, action: int):
        if self.last_state is None or self.last_action is None:
            return

        
        
        # Terminal detection uses the raw reward; scaling happens afterwards.
        done = self._is_terminal_transition(state, reward)

        clipped_reward = float(reward) / self.reward_scale
        if self.reward_clip is not None:
            clipped_reward = float(np.clip(clipped_reward, -self.reward_clip, self.reward_clip))
        self.states.append(self.last_state)
        self.actions.append(self.last_action)
        self.rewards.append(clipped_reward)
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

    def _mark_boundary(self, bootstrap_state: tuple[float, ...]):
        """Cut the GAE chain after the last stored transition, bootstrapping
        with V(bootstrap_state). No-op if the chain is already cut."""
        if self.dones and not self.dones[-1]:
            self.dones[-1] = True
            self.next_values[-1] = self._value(bootstrap_state)

    def finish_rollout(self, bootstrap_state: tuple[float, ...] | None = None):
        """Mark an episode boundary and flush the buffer when appropriate.

        With `bootstrap_state` (episode truncated by a step limit): record the
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

        if self.replay_buffer is not None:
            self.replay_buffer.push_batch(states, actions, old_log_probs, adv_tensor, ret_tensor)

        self._optimize(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=adv_tensor,
            returns=ret_tensor,
        )
        self._clear_rollout()

    def _optimize(
        self,
        states: list[tuple[float, ...]],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ):
        n = len(states)
        batch_size = self.minibatch_size or n
        batch_size = max(1, min(batch_size, n))

        use_replay = (
            self.replay_buffer is not None
            and len(self.replay_buffer) >= batch_size
        )

        for _ in range(self.update_epochs):
            if use_replay:
                # Off-policy: sample a batch of size n from the full replay
                # buffer. IS correction is handled by the existing ratio term.
                b_states, b_actions, b_log_probs, b_advantages, b_returns = \
                    self.replay_buffer.sample(n, self.device)
            else:
                b_states, b_actions, b_log_probs, b_advantages, b_returns = \
                    states, actions, old_log_probs, advantages, returns

            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_states = [b_states[int(i.item())] for i in batch_idx]

                dist = self._distribution(batch_states)
                new_log_probs = dist.log_prob(b_actions[batch_idx])
                entropy = dist.entropy().mean()
                values = self._critic_values(self._state_tensor(batch_states)).squeeze(-1)

                ratios = torch.exp(new_log_probs - b_log_probs[batch_idx])
                unclipped = ratios * b_advantages[batch_idx]
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * b_advantages[batch_idx]

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(values, b_returns[batch_idx])
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                if self.network_mode == "shared":
                    self.optimizer.zero_grad()
                else:
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm is not None:
                    if self.network_mode == "shared":
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    else:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                if self.network_mode == "shared":
                    self.optimizer.step()
                else:
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
        if self.network_mode == "shared":
            self.model.train(enabled)
        else:
            self.actor.train(enabled)
            self.critic.train(enabled)

    def policy(self, state: tuple[float, ...]) -> np.ndarray:
        """Return the actor network's action probabilities for `state`."""
        with torch.no_grad():
            dist = self._distribution(state)
        return dist.probs.squeeze(0).detach().cpu().numpy()
