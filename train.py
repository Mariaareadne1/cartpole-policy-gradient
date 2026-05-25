"""
Robot Reacher — Policy Gradient (REINFORCE) from scratch
Maria Showalter · Columbia CS

A 2-joint robot arm learns to reach a target position using RL.
The policy network is a small MLP in PyTorch — no stable-baselines, no RL libraries.
We implement the REINFORCE algorithm manually.

Environment: custom 2-DOF arm sim (no Gym dependency required to understand the RL logic)
Optional: swap in gymnasium's Reacher-v4 by changing the env lines at the bottom.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path


# ─── Environment ──────────────────────────────────────────────────────────────

class ReacherEnv:
    """
    2-joint planar robot arm trying to reach a target.
    State:  [cos(θ1), sin(θ1), cos(θ2), sin(θ2), target_x, target_y,
             fingertip_x - target_x, fingertip_y - target_y]  → 8 dims
    Action: [dθ1, dθ2]  → 2 continuous torques, clipped to [-1, 1]
    Reward: -dist(fingertip, target)  — dense reward
    """

    def __init__(self):
        self.l1 = 1.0   # link 1 length
        self.l2 = 0.8   # link 2 length
        self.dt = 0.05
        self.max_steps = 200
        self.obs_dim = 8
        self.act_dim = 2

    def reset(self):
        self.theta1 = np.random.uniform(-np.pi, np.pi)
        self.theta2 = np.random.uniform(-np.pi, np.pi)
        self.target = np.random.uniform(-1.5, 1.5, size=2)
        # Ensure target is reachable
        while np.linalg.norm(self.target) > self.l1 + self.l2 - 0.1:
            self.target = np.random.uniform(-1.5, 1.5, size=2)
        self.step_count = 0
        return self._obs()

    def _fingertip(self):
        x = self.l1 * np.cos(self.theta1) + self.l2 * np.cos(self.theta1 + self.theta2)
        y = self.l1 * np.sin(self.theta1) + self.l2 * np.sin(self.theta1 + self.theta2)
        return np.array([x, y])

    def _obs(self):
        fp = self._fingertip()
        return np.array([
            np.cos(self.theta1), np.sin(self.theta1),
            np.cos(self.theta2), np.sin(self.theta2),
            self.target[0], self.target[1],
            fp[0] - self.target[0], fp[1] - self.target[1],
        ], dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.theta1 += self.dt * action[0] * 3.0
        self.theta2 += self.dt * action[1] * 3.0
        self.step_count += 1

        fp = self._fingertip()
        dist = np.linalg.norm(fp - self.target)
        reward = -dist  # dense: closer = better

        done = (self.step_count >= self.max_steps)
        return self._obs(), reward, done, {"dist": dist}

    def render_state(self):
        """Return dict for visualization."""
        fp = self._fingertip()
        elbow = np.array([
            self.l1 * np.cos(self.theta1),
            self.l1 * np.sin(self.theta1),
        ])
        return {
            "theta1": float(self.theta1), "theta2": float(self.theta2),
            "elbow": elbow.tolist(), "fingertip": fp.tolist(),
            "target": self.target.tolist(),
        }


# ─── Policy Network ───────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Gaussian policy: outputs mean + log_std for each action dimension.
    We parameterize log_std as a learned parameter (not state-dependent) for simplicity.

    Architecture: obs → Linear(64) → Tanh → Linear(64) → Tanh → Linear(act_dim)
    """

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # Learnable log standard deviation (shared across states)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        std  = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def select_action(self, obs_np):
        """Sample action + compute log probability (needed for REINFORCE)."""
        obs = torch.FloatTensor(obs_np).unsqueeze(0)
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).numpy(), log_prob


# ─── REINFORCE (Policy Gradient) ─────────────────────────────────────────────

def compute_returns(rewards, gamma):
    """
    Discounted returns: G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    Computed backwards for efficiency.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    # Normalize returns (reduces variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train(
    num_episodes=2000,
    gamma=0.99,
    lr=3e-4,
    log_every=50,
    save_path="checkpoints/policy.pt",
):
    env    = ReacherEnv()
    policy = PolicyNetwork(env.obs_dim, env.act_dim)
    opt    = optim.Adam(policy.parameters(), lr=lr)

    Path("checkpoints").mkdir(exist_ok=True)
    log = []

    for ep in range(num_episodes):
        obs = env.reset()
        rewards, log_probs = [], []

        done = False
        while not done:
            action, log_prob = policy.select_action(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)

        # ── REINFORCE update ──────────────────────────────────────────────────
        # Loss = -E[G_t · log π(a_t|s_t)]
        # Gradient ascent on expected return ≡ gradient descent on negative loss
        returns   = compute_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        loss      = -(log_probs * returns).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        opt.step()
        # ─────────────────────────────────────────────────────────────────────

        ep_return = sum(rewards)
        final_dist = info["dist"]
        log.append({"ep": ep, "return": ep_return, "dist": final_dist})

        if ep % log_every == 0:
            recent = log[-log_every:]
            avg_ret  = np.mean([x["return"] for x in recent])
            avg_dist = np.mean([x["dist"]   for x in recent])
            print(f"  ep {ep:4d} | avg return {avg_ret:6.1f} | avg final dist {avg_dist:.3f}")

    torch.save(policy.state_dict(), save_path)
    print(f"\nSaved policy → {save_path}")

    with open("training_log.json", "w") as f:
        json.dump(log, f)
    print("Saved training_log.json")

    return policy, log


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(policy_path="checkpoints/policy.pt", n_episodes=20):
    env    = ReacherEnv()
    policy = PolicyNetwork(env.obs_dim, env.act_dim)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    dists = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _ = policy.select_action(obs)
                obs, _, done, info = env.step(action)
            dists.append(info["dist"])

    print(f"Eval over {n_episodes} episodes:")
    print(f"  Mean final dist to target: {np.mean(dists):.4f}")
    print(f"  Reached within 0.1:  {sum(d < 0.1 for d in dists)}/{n_episodes} episodes")
    return dists


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--eval",     action="store_true")
    args = p.parse_args()

    if args.eval:
        evaluate()
    else:
        print("Training REINFORCE policy on 2-DOF reacher...")
        print("─" * 50)
        policy, log = train(num_episodes=args.episodes)
        print("─" * 50)
        print("Training complete. Run with --eval to test.")
