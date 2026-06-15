"""Module M6: tabular Q-learning for adaptive economic policy.

The exam asks for a simplified Markov Decision Process (MDP) where the state is
composed of GDP growth, digital readiness, AI capacity, and unemployment risk.
This module avoids an external gymnasium dependency so that the repository runs
locally after a basic pip install.  The API mirrors a gym environment: reset,
step, and train_q_learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ACTIONS = {
    0: np.array([0.70, 0.10, 0.10, 0.10]),
    1: np.array([0.40, 0.25, 0.15, 0.20]),
    2: np.array([0.25, 0.45, 0.15, 0.15]),
    3: np.array([0.20, 0.20, 0.45, 0.15]),
    4: np.array([0.30, 0.20, 0.10, 0.40]),
}

ACTION_NAMES = {
    0: "Truyền thống",
    1: "Cân bằng",
    2: "Số hóa nhanh",
    3: "AI dẫn dắt",
    4: "Bao trùm",
}


@dataclass
class EconomyState:
    """Continuous economic state used inside the simple environment."""

    capital: float = 27500.0
    digital: float = 20.3
    ai: float = 86.0
    human: float = 30.0
    labor: float = 54.0
    t: int = 0


class VietnamEconomyEnv:
    """Minimal MDP environment for policy-allocation experiments."""

    def __init__(self, horizon: int = 10, annual_budget: float = 1000.0, seed: int | None = None):
        self.horizon = horizon
        self.annual_budget = annual_budget
        self.rng = np.random.default_rng(seed)
        self.state = EconomyState()
        self.discrete_state = np.array([1, 1, 0, 1], dtype=int)

    def reset(self) -> np.ndarray:
        """Reset to the Vietnam 2026 baseline state."""
        self.state = EconomyState()
        self.discrete_state = np.array([1, 1, 0, 1], dtype=int)
        return self.discrete_state.copy()

    @staticmethod
    def production(state: EconomyState) -> float:
        """Evaluate a simple Cobb-Douglas production value."""
        return state.capital**0.33 * state.labor**0.42 * state.digital**0.10 * state.ai**0.08 * state.human**0.07

    @staticmethod
    def discretize(growth: float, digital: float, ai: float, unemployment: float) -> np.ndarray:
        """Discretize continuous indicators to four 3-level state components."""
        g = 0 if growth < 0.055 else 1 if growth < 0.075 else 2
        d = 0 if digital < 22 else 1 if digital < 28 else 2
        a = 0 if ai < 95 else 1 if ai < 120 else 2
        u = 0 if unemployment < 0.04 else 1 if unemployment < 0.07 else 2
        return np.array([g, d, a, u], dtype=int)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Apply an allocation action and return next state, reward, done, info."""
        allocation = ACTIONS[int(action)]
        old_output = self.production(self.state)
        self.state.capital += allocation[0] * self.annual_budget
        self.state.digital += allocation[1] * self.annual_budget / 100.0
        self.state.ai += allocation[2] * self.annual_budget / 20.0
        self.state.human += allocation[3] * self.annual_budget / 200.0
        self.state.labor *= 1.006
        new_output = self.production(self.state)
        growth = (new_output - old_output) / old_output
        unemployment = max(0.025, 0.075 + 0.05 * allocation[2] - 0.04 * allocation[3] - 0.5 * (growth - 0.06))
        cyber_risk = 0.02 + 0.12 * allocation[2] - 0.03 * allocation[3]
        emission = 0.03 + 0.10 * allocation[0] + 0.08 * allocation[2] - 0.02 * allocation[3]
        reward = 0.40 * growth * 100.0 - 0.25 * unemployment * 100.0 - 0.20 * cyber_risk * 100.0 - 0.15 * emission * 100.0
        self.state.t += 1
        done = self.state.t >= self.horizon
        self.discrete_state = self.discretize(growth, self.state.digital, self.state.ai, unemployment)
        info = {
            "growth": float(growth),
            "output": float(new_output),
            "unemployment": float(unemployment),
            "cyber_risk": float(cyber_risk),
            "emission": float(emission),
        }
        return self.discrete_state.copy(), float(reward), done, info


def q_table_shape() -> Tuple[int, int, int, int, int]:
    """Return the fixed tabular Q-learning tensor shape."""
    return (3, 3, 3, 3, 5)


def epsilon_schedule(episode: int, total_episodes: int, eps_min: float = 0.05) -> float:
    """Linear decay schedule for epsilon-greedy exploration."""
    half = max(1, total_episodes // 2)
    return max(eps_min, 1.0 - episode / half)


def choose_action(q: np.ndarray, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Choose an action according to epsilon-greedy policy."""
    if rng.random() < epsilon:
        return int(rng.integers(5))
    return int(np.argmax(q[tuple(state)]))


def train_q_learning(
    episodes: int = 5000,
    learning_rate: float = 0.10,
    discount: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Train a tabular Q-learning policy."""
    rng = np.random.default_rng(seed)
    q = np.zeros(q_table_shape())
    history = []
    env = VietnamEconomyEnv(seed=seed)
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(env.horizon):
            epsilon = epsilon_schedule(episode, episodes)
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            old = q[tuple(state) + (action,)]
            target = reward + discount * q[tuple(next_state)].max()
            q[tuple(state) + (action,)] = old + learning_rate * (target - old)
            total_reward += reward
            state = next_state
            if done:
                break
        if episode % max(1, episodes // 200) == 0 or episode == episodes - 1:
            history.append({"episode": episode, "total_reward": total_reward, "epsilon": epsilon_schedule(episode, episodes)})
    return q, pd.DataFrame(history)


def extract_policy(q: np.ndarray) -> pd.DataFrame:
    """Extract argmax action for every discrete state."""
    rows = []
    for g in range(3):
        for d in range(3):
            for ai in range(3):
                for u in range(3):
                    action = int(np.argmax(q[g, d, ai, u]))
                    rows.append(
                        {
                            "growth_state": g,
                            "digital_state": d,
                            "ai_state": ai,
                            "unemployment_state": u,
                            "action_id": action,
                            "action_name": ACTION_NAMES[action],
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_policy(policy_action: int | None = None, q: np.ndarray | None = None, episodes: int = 200, seed: int = 7) -> pd.DataFrame:
    """Evaluate a fixed policy or a learned Q-table policy."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        env = VietnamEconomyEnv(seed=int(rng.integers(1_000_000)))
        state = env.reset()
        total_reward = 0.0
        for _ in range(env.horizon):
            if q is not None:
                action = int(np.argmax(q[tuple(state)]))
            elif policy_action is not None:
                action = int(policy_action)
            else:
                action = int(rng.integers(5))
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        rows.append({"episode": ep, "total_reward": total_reward})
    return pd.DataFrame(rows)


def compare_rule_based_policies(q: np.ndarray | None = None) -> pd.DataFrame:
    """Compare learned policy with rule-based alternatives."""
    frames = []
    if q is not None:
        learned = evaluate_policy(q=q)
        learned["policy"] = "Q-learning"
        frames.append(learned)
    for action in [1, 3, 4]:
        df = evaluate_policy(policy_action=action)
        df["policy"] = f"Always {ACTION_NAMES[action]}"
        frames.append(df)
    random_df = evaluate_policy(policy_action=None, q=None)
    random_df["policy"] = "Random"
    frames.append(random_df)
    return pd.concat(frames, ignore_index=True)


def action_recommendations(q: np.ndarray, states: List[Tuple[int, int, int, int]] | None = None) -> pd.DataFrame:
    """Return recommended actions for selected states."""
    if states is None:
        states = [(1, 1, 0, 1), (0, 0, 0, 2), (2, 2, 2, 0), (1, 0, 1, 2), (0, 2, 2, 1)]
    rows = []
    for state in states:
        action = int(np.argmax(q[state]))
        rows.append({"state": str(state), "action_id": action, "action_name": ACTION_NAMES[action]})
    return pd.DataFrame(rows)


def run_m6(episodes: int = 3000) -> Dict[str, object]:
    """Run Q-learning and return policy artifacts."""
    q, history = train_q_learning(episodes=episodes)
    return {
        "q": q,
        "history": history,
        "policy_table": extract_policy(q),
        "recommendations": action_recommendations(q),
        "evaluation": compare_rule_based_policies(q),
    }
