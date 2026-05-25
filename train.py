"""
CartPole — Policy Gradient (REINFORCE) from scratch
Maria Showalter · Columbia CS

Trains a neural network policy in PyTorch to balance a pole on a cart.
No stable-baselines, no RL libraries — just PyTorch + Gymnasium.

CartPole is the standard RL benchmark: the agent must learn to push the cart
left/right to keep the pole upright. Solved = avg reward >= 475 over 100 episodes.

Key implementation detail: batch REINFORCE — collect N episodes before each
gradient update, normalize returns across the whole batch. This reduces variance
enough for reliable convergence (single-episode normalization is too noisy).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path

try:
    import gymnasium as gym
except ImportError:
    import gym


# ─── Policy Network ───────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Discrete policy for CartPole.
    State (4D): [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    Output: categorical distribution over [push-left, push-right]
    """
    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
            nn.Linear(hidden,  act_dim),
        )

    def forward(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))

    def select_action(self, obs_np):
        obs  = torch.FloatTensor(obs_np)
        dist = self(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# ─── REINFORCE ────────────────────────────────────────────────────────────────

def compute_returns(rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def train(num_episodes=600, gamma=0.99, lr=1e-2, batch_size=5,
          log_every=50, save_path="checkpoints/policy.pt"):

    env    = gym.make("CartPole-v1")
    policy = PolicyNetwork()
    opt    = optim.Adam(policy.parameters(), lr=lr)

    Path("checkpoints").mkdir(exist_ok=True)
    log        = []
    reward_buf = []
    batch      = []   # accumulate episodes before updating

    for ep in range(num_episodes):
        obs, _ = env.reset()
        rewards, log_probs = [], []
        done = False

        while not done:
            action, log_prob = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)

        returns = compute_returns(rewards, gamma)
        batch.append((torch.stack(log_probs), returns))

        ep_return = sum(rewards)
        reward_buf.append(ep_return)
        log.append({"ep": ep, "return": float(ep_return)})

        # ── Update once we have a full batch ─────────────────────────────────
        if len(batch) == batch_size:
            # Normalize returns across ALL episodes in the batch (key for stability)
            all_returns = torch.cat([r for _, r in batch])
            mean = all_returns.mean()
            std  = all_returns.std() + 1e-8

            loss = torch.stack([
                -(lps * (rets - mean) / std).mean()
                for lps, rets in batch
            ]).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            batch = []
        # ─────────────────────────────────────────────────────────────────────

        if ep % log_every == 0:
            avg = np.mean(reward_buf[-log_every:])
            bar = "█" * int(avg / 20)
            print(f"  ep {ep:4d} | avg return {avg:6.1f}  {bar}")

        if len(reward_buf) >= 100 and np.mean(reward_buf[-100:]) >= 475:
            print(f"\n  ✓ Solved at episode {ep}! (avg >= 475 over last 100 eps)")
            break

    env.close()
    torch.save(policy.state_dict(), save_path)
    print(f"\nSaved policy → {save_path}")

    with open("training_log.json", "w") as f:
        json.dump(log, f)
    print("Saved training_log.json")

    return policy, log


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(policy_path="checkpoints/policy.pt", n_episodes=20):
    env    = gym.make("CartPole-v1")
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(policy_path, weights_only=True))
    policy.eval()

    returns = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False; total = 0
            while not done:
                dist   = policy(torch.FloatTensor(obs))
                action = dist.probs.argmax().item()   # greedy at eval time
                obs, r, term, trunc, _ = env.step(action)
                done   = term or trunc
                total += r
            returns.append(total)

    env.close()
    print(f"\nEval over {n_episodes} episodes:")
    print(f"  Mean return:   {np.mean(returns):.1f}  (max 500)")
    print(f"  Solved (≥475): {sum(r >= 475 for r in returns)}/{n_episodes} episodes")
    return returns


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--eval", action="store_true")
    args = p.parse_args()

    if args.eval:
        evaluate()
    else:
        print("Training REINFORCE on CartPole-v1...")
        print("─" * 52)
        train(num_episodes=args.episodes)
        print("─" * 52)
        evaluate()
