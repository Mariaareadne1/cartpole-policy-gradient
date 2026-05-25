# rl-reacher

A 2-joint robot arm trained to reach a target using policy gradient reinforcement learning — implemented from scratch in PyTorch.

No stable-baselines, no RL libraries. The training loop, policy network, and REINFORCE algorithm are all written by hand.

---

## What it does

A simulated planar robot arm with two joints learns, through trial and error, to move its fingertip to a randomly-placed target. The policy — a neural network that maps arm state to joint torques — is trained with the REINFORCE algorithm.

**State (8D):** joint angles (as cos/sin pairs), target position, fingertip-to-target error vector  
**Action (2D):** continuous torques for each joint, clipped to [-1, 1]  
**Reward:** `-distance(fingertip, target)` — dense, so the arm gets a gradient signal every step

---

## The RL implementation

```python
# REINFORCE in one paragraph:
# 1. Run episode, collect (state, action, reward) trajectory
# 2. Compute discounted returns G_t = r_t + γ·r_{t+1} + ...
# 3. Normalize returns (variance reduction)
# 4. Update: maximize E[G_t · log π(a_t | s_t)]

loss = -(log_probs * returns).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The policy is a Gaussian: the network outputs a mean action, and a learned `log_std` parameter controls exploration. As training progresses, `log_std` decreases — the arm becomes more deliberate.

---

## Run it

```bash
pip install torch numpy
python train.py --episodes 2000
```

Logs to `training_log.json`. To evaluate a saved policy:

```bash
python train.py --eval
```

Expected output after ~1000 episodes:
```
Eval over 20 episodes:
  Mean final dist to target: 0.087
  Reached within 0.1:  17/20 episodes
```

---

## Browser visualizer

Open `visualizer.html` — no server needed. Shows the arm learning in real time:

- Watch early episodes: random flailing
- Watch episodes 200–400: the arm starts orienting toward the target
- Watch 600+: consistent reaching

The link color shifts from red → green as the fingertip approaches the target.

---

## Architecture

```
PolicyNetwork:
  Linear(8, 64) → Tanh
  Linear(64, 64) → Tanh
  Linear(64, 2)       → mean actions

  log_std             → learnable parameter (shared)
```

Small on purpose: the environment is low-dimensional. Bigger networks don't help here and make the training dynamics harder to interpret.

---

## Why I built this

I've spent a lot of time building systems where emergent behavior comes from simple rules — cellular automata, generative music, live coding. RL felt like the same idea applied to physical control: you don't program the behavior, you specify what counts as good, and the system figures out the rest.

The reacher is a clean case of that. The reward function is one line. The behavior that emerges — a two-joint arm learning coordinated movement to reach arbitrary targets — is genuinely interesting to watch happen.

---

## Files

```
train.py          — environment, policy network, REINFORCE loop (PyTorch)
visualizer.html   — browser-based training visualization
checkpoints/      — saved policy weights (after training)
training_log.json — episode returns (plot with tensorboard or matplotlib)
```

---

*Maria Showalter · Columbia CS · [mariashowalter.com](https://mariashowalter.com)*
