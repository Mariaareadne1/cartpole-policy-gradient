# cartpole-policy-gradient

REINFORCE policy gradient implemented from scratch in PyTorch, trained on CartPole-v1.

**Result: solved in 288 episodes. 20/20 perfect episodes at eval (mean return 500/500).**

---

## What it does

A neural network learns to balance a pole on a cart by deciding which direction to push — trained purely through trial and error, with no hand-coded rules.

The policy, training loop, and return computation are all written by hand. No stable-baselines, no RL libraries — just PyTorch and Gymnasium.

```
Training REINFORCE on CartPole-v1...
  ep    0 | avg return   15.0
  ep   50 | avg return   72.8  ███
  ep  100 | avg return  212.9  ██████████
  ep  150 | avg return  292.8  ██████████████
  ep  200 | avg return  396.6  ███████████████████
  ep  250 | avg return  484.6  ████████████████████████
  ✓ Solved at episode 288!

Eval over 20 episodes:
  Mean return:   500.0  (max 500)
  Solved (≥475): 20/20 episodes
```

---

## The algorithm

REINFORCE: run an episode, collect rewards, update the policy to make good actions more likely.

```python
# Core update — the whole algorithm in four lines:
returns   = compute_discounted_returns(rewards, gamma=0.99)
log_probs = torch.stack(log_probs)
loss      = -(log_probs * returns).mean()
loss.backward()
```

**Key implementation detail — batch REINFORCE:**
Single-episode return normalization destroys the learning signal (every episode gets normalized to mean=0, std=1 regardless of quality). Instead, I collect 5 episodes per update and normalize returns across the whole batch. This was the difference between flat training curves and convergence.

---

## Policy network

```
Input (4D): [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

Linear(4 → 64) → Tanh
Linear(64 → 64) → Tanh
Linear(64 → 2)  → Categorical distribution over [push-left, push-right]
```

At training time: sample from the distribution (exploration).  
At eval time: take the argmax (greedy).

---

## Run it

```bash
pip install torch gymnasium
python train.py
```

Trains in ~2 minutes. To evaluate a saved checkpoint:

```bash
python train.py --eval
```

---

## Why I built this

I've been building systems where behavior emerges from rules — cellular automata, generative music, live coding. RL is the same idea applied to physical control: instead of specifying the behavior, you specify what counts as good, and the system figures out the rest.

CartPole and the [PID inverted pendulum](https://github.com/Mariaareadne1/inverted-pendulum-pid) are the same physical problem solved two ways — one with a hand-tuned feedback controller, one with a learned policy. Comparing them is interesting: PID is faster to tune and perfectly interpretable; RL generalizes but needs thousands of rollouts to find what PID gets analytically.

---

## Files

```
train.py              — policy network + REINFORCE loop (PyTorch)
visualizer.html       — browser visualization of the training curve
checkpoints/policy.pt — saved weights after training
training_log.json     — episode returns (for plotting)
```

---

*Maria Showalter · Columbia CS · [mariashowalter.com](https://mariashowalter.com)*
