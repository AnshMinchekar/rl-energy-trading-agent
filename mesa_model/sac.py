# -*- coding: utf-8 -*-
"""
Soft Actor-Critic (SAC) for the storage arbitrage agent.

This module is self-contained and PyTorch-based. It is wired into the
``storage`` agent in ``mesa_model/agents.py`` via the ``SACLearner`` class.

Design (see docs/redesign-plan.md):
  * Squashed-Gaussian actor (reparameterised) → action ∈ (-1, 1)
  * Twin Q-critics + Polyak-averaged target critics (min-target → no overestimation)
  * Automatic entropy temperature α (target_entropy = -1 for a scalar action)
  * Uniform-sampled experience replay; random warm-up to seed the buffer

Phase 1: ReplayBuffer + twin Critic networks + critic update path.
Phase 2: GaussianActor, automatic α tuning, and the full SACLearner update loop.
"""

import os
# PyTorch bundles its own OpenMP (libiomp5md), which clashes with the MKL
# OpenMP already loaded by numpy/gurobi on Windows and aborts the process.
# Allow the duplicate and keep torch single-threaded (the nets are tiny, and
# this avoids oversubscribing the gurobi market solve). Must precede import torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Experience replay
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity circular buffer backed by preallocated numpy arrays."""

    def __init__(self, capacity, state_dim, action_dim=1, seed=42):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.RandomState(seed)

    def push(self, state, action, reward, next_state, done):
        i = self._ptr
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self._rng.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._states[idx]),
            torch.from_numpy(self._actions[idx]),
            torch.from_numpy(self._rewards[idx]),
            torch.from_numpy(self._next_states[idx]),
            torch.from_numpy(self._dones[idx]),
        )

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """Q(s, a): (state_dim + action_dim) → hidden → hidden → 1."""

    def __init__(self, state_dim, action_dim=1, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class GaussianActor(nn.Module):
    """Squashed-Gaussian policy.

    Forward produces (mu, log_std) of a Normal over the pre-squash variable u;
    the action is tanh(u) ∈ (-1, 1). ``sample`` uses the reparameterisation
    trick so the entropy term is differentiable, and applies the tanh log-prob
    correction term ``-Σ log(1 - tanh(u)^2)``.
    """

    def __init__(self, state_dim, action_dim=1, hidden=64):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.n1 = nn.LayerNorm(hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = F.relu(self.n1(self.l1(state)))
        h = F.relu(self.n2(self.l2(h)))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """Return (action, log_prob, deterministic_action)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()                       # reparameterised
        action = torch.tanh(u)
        # tanh change-of-variables correction (sum over action dims)
        log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


# ---------------------------------------------------------------------------
# SAC learner
# ---------------------------------------------------------------------------
class SACLearner:
    """Owns the actor, twin critics + targets, the temperature α, and replay.

    Public API used by the storage agent:
        select_action(state, deterministic) -> float in [-1, 1]
        push(state, action, reward, next_state, done)
        update() -> dict of diagnostics (or None if still warming up)
    """

    def __init__(self, state_dim, action_dim=1, hidden=64,
                 gamma=0.99, tau=0.005, lr=3e-4, target_entropy=-1.0,
                 buffer_size=100_000, batch_size=256, warmup_steps=2_000,
                 actor_update_every=2, reward_scale=10.0, seed=42, device="cpu"):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.actor_update_every = actor_update_every
        self.reward_scale = reward_scale
        self.target_entropy = target_entropy
        self.action_dim = action_dim

        self.actor = GaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.q1 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q1_t = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2_t = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        for p in self.q1_t.parameters():
            p.requires_grad_(False)
        for p in self.q2_t.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # Temperature: optimise log_alpha for positivity of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, seed=seed)
        self._rng = np.random.RandomState(seed + 1)
        self.total_env_steps = 0   # transitions observed
        self.total_updates = 0     # gradient updates performed

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # -- acting -------------------------------------------------------------
    def select_action(self, state, deterministic=False):
        """state: 1-D array-like. Returns a python float in [-1, 1]."""
        if (not deterministic) and self.total_env_steps < self.warmup_steps:
            return float(self._rng.uniform(-1.0, 1.0))
        s = torch.as_tensor(np.asarray(state, dtype=np.float32),
                            device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.sample(s)
        a = mean if deterministic else action
        return float(a.squeeze().item())

    # -- storing ------------------------------------------------------------
    def push(self, state, action, reward, next_state, done):
        self.buffer.push(
            np.asarray(state, dtype=np.float32),
            np.asarray([action], dtype=np.float32),
            np.float32(reward),
            np.asarray(next_state, dtype=np.float32),
            np.float32(done),
        )
        self.total_env_steps += 1

    # -- learning -----------------------------------------------------------
    def update(self):
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None

        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s, a, r, s2, d = (t.to(self.device) for t in (s, a, r, s2, d))
        r = r * self.reward_scale

        alpha = self.log_alpha.exp().detach()

        # --- Critic update ---
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(s2)
            min_q_t = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2))
            target = r + self.gamma * (1.0 - d) * (min_q_t - alpha * logp2)
        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)
        critic_loss = q1_loss + q2_loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        diagnostics = {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(target.mean().item()),
            "alpha": float(alpha.item()),
        }

        # --- Actor + temperature update (delayed) ---
        if self.total_updates % self.actor_update_every == 0:
            a_new, logp, _ = self.actor.sample(s)
            q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
            actor_loss = (alpha * logp - q_new).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Polyak update of target critics
            self._soft_update(self.q1, self.q1_t)
            self._soft_update(self.q2, self.q2_t)

            diagnostics["actor_loss"] = float(actor_loss.item())
            diagnostics["entropy"] = float((-logp).mean().item())

        self.total_updates += 1
        return diagnostics

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

    # -- persistence --------------------------------------------------------
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_t": self.q1_t.state_dict(),
            "q2_t": self.q2_t.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "total_env_steps": self.total_env_steps,
            "total_updates": self.total_updates,
        }


# ---------------------------------------------------------------------------
# Self-test (Phase 1): exercise the buffer + a single critic update step
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    state_dim = 16
    buf = ReplayBuffer(capacity=1000, state_dim=state_dim)

    # Seed the buffer with random transitions
    for _ in range(500):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.uniform(-1, 1, size=1).astype(np.float32)
        r = np.float32(np.random.randn())
        s2 = np.random.randn(state_dim).astype(np.float32)
        buf.push(s, a, r, s2, 0.0)
    assert len(buf) == 500, f"expected 500, got {len(buf)}"

    # Twin critics + targets
    q1, q2 = Critic(state_dim), Critic(state_dim)
    q1_t, q2_t = Critic(state_dim), Critic(state_dim)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)

    gamma = 0.99
    s, a, r, s2, d = buf.sample(256)

    # Placeholder next-action (Phase 2 replaces with actor sample); checks the
    # min-target Bellman backup and that gradients flow through both critics.
    with torch.no_grad():
        a2 = torch.empty_like(a).uniform_(-1, 1)
        min_q = torch.min(q1_t(s2, a2), q2_t(s2, a2))
        y = r + gamma * (1.0 - d) * min_q

    loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
    before = loss.item()
    opt.zero_grad()
    loss.backward()
    # confirm gradients exist
    assert q1.l1.weight.grad is not None and q2.l3.bias.grad is not None
    opt.step()

    # Loss should drop after one step on the same batch
    after = (F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)).item()
    assert after < before, f"critic loss did not decrease: {before} -> {after}"
    print(f"[sac self-test] buffer OK (len={len(buf)}); "
          f"critic loss {before:.4f} -> {after:.4f}  PASS")


def _self_test_learner():
    """Phase 2: full SACLearner on a toy 1-D arbitrage MDP.

    Toy task: state = [signal, holding]. A hidden 'signal' (±1) says whether
    the next reward favours holding +1 or -1. Reward = action * signal. The
    optimal policy is action = sign(signal). We check the learner's greedy
    action aligns with the signal far above chance after training.
    """
    rng = np.random.RandomState(0)
    state_dim = 2
    learner = SACLearner(state_dim, hidden=32, warmup_steps=500,
                        batch_size=128, buffer_size=20_000, seed=1)

    def sample_env():
        signal = 1.0 if rng.rand() < 0.5 else -1.0
        s = np.array([signal, rng.uniform(-1, 1)], dtype=np.float32)
        return s, signal

    # Interact + learn
    for step in range(6000):
        s, signal = sample_env()
        a = learner.select_action(s, deterministic=False)
        reward = a * signal                       # maximised by a = sign(signal)
        s2, _ = sample_env()
        learner.push(s, a, reward, s2, 0.0)
        learner.update()

    # Evaluate greedy policy alignment with the signal
    correct = 0
    n_eval = 400
    for _ in range(n_eval):
        s, signal = sample_env()
        a = learner.select_action(s, deterministic=True)
        if np.sign(a) == np.sign(signal):
            correct += 1
    acc = correct / n_eval
    assert acc > 0.85, f"learner failed to solve toy MDP: acc={acc:.2f}"
    print(f"[sac self-test] learner toy-MDP greedy accuracy = {acc:.2%}  "
          f"(alpha={learner.alpha.item():.3f})  PASS")


if __name__ == "__main__":
    _self_test()
    _self_test_learner()
