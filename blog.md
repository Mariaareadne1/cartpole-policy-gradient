# What it took to get a pole to stop falling

*May 2026 · RL · PyTorch · Building*

---

I built this in a day. Not because I've been dreaming about robotics — I'm applying for a robotics internship and wanted something real to show. That's the honest version.

But somewhere between the third failed training run and watching the reward finally climb, something genuinely clicked. So this is about both things: the strategic build, and what I actually learned.

---

## The problem

CartPole is simple to describe. A pole is balanced on a cart. Every timestep, you push the cart left or right. The pole falls if you don't. You get one point for every step the pole stays up. Maximum score: 500. "Solved" means averaging 475 over 100 episodes.

A human can do it in a few seconds. Teaching a neural network to do it is a classic benchmark in reinforcement learning — not because it's hard, but because it's clean. The environment is simple enough that if your algorithm is wrong, you know immediately.

Mine was wrong. Twice.

---

## Three attempts

**Attempt 1:** I started with a custom 2-joint robot arm environment — more visually interesting, more robotics-adjacent. REINFORCE with a value baseline. 3000 episodes. The average final distance to target barely moved. Flatlined at 1.5 the entire run.

**Attempt 2:** Switched to CartPole, which has a well-tested reward signal. Still flatlined. Mean return stuck around 25 through 1200 episodes.

Both times the issue wasn't the network architecture or the learning rate. It was something subtler about how REINFORCE handles variance.

**Attempt 3:** Worked. Solved at episode 288. 20/20 perfect episodes at eval.

---

## What actually fixed it

REINFORCE is conceptually simple:

```
run an episode
compute discounted returns G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
update the policy: make actions that led to high returns more likely
```

The math is clean. The implementation detail that kills it is variance.

Each episode's returns get normalized — subtracted by mean, divided by std — to make training stable. The problem: if you normalize *per episode*, every episode looks the same to the optimizer. A terrible episode where the pole falls in 15 steps and a great one where it survives 400 steps both normalize to mean=0, std=1. The signal is gone.

The fix was **batch REINFORCE**: collect 5 episodes, then normalize returns across all 5 together. Now the optimizer can actually tell the difference between a bad episode and a good one. That's it. That's what two failed runs were missing.

```python
# Wrong: normalize each episode independently
returns = (returns - returns.mean()) / returns.std()

# Right: normalize across the whole batch
all_returns = torch.cat([r for _, r in batch])
mean, std = all_returns.mean(), all_returns.std()
advantages = (returns - mean) / std
```

Watching it converge after that was mostly just relief. But the relief came with actual understanding — I now know exactly why REINFORCE fails, not just that it can.

---

## Why this connects to things I already think about

I've spent the last year building generative music systems. Cellular automata that compose patterns. A synthesizer that does its own music theory. Live coding performances where the code runs in real time while the audience watches.

What those have in common: you don't specify the output. You specify the rules, or the reward, or the constraints — and the behavior emerges. A cellular automaton generates music no one wrote. An RL agent learns to balance a pole without being told how.

That's not a stretch. It's genuinely how I think about both.

The difference is that in RL, the emergence is *trained* — the system improves through feedback. In cellular automata, it's fixed rules producing complex behavior. RL adds the loop: act, observe, update. It's a more interesting kind of emergence because it's directed.

---

## The two-repo thing

I also built a [PID controller](https://github.com/Mariaareadne1/inverted-pendulum-pid) the same week — an inverted pendulum stabilized by a hand-tuned feedback controller rather than a learned policy.

It's the same physical problem. The PID controller uses the math directly: measure the angle, compute the correction analytically. The RL agent figures out what to do by trying things thousands of times and learning from the outcomes.

PID converges instantly and is perfectly interpretable. RL takes 288 episodes and is a black box. Neither is better — they're different tools, and understanding why you'd reach for one versus the other is more interesting than the implementations themselves.

---

## On applying strategically

I said at the start that I built this strategically. That's true. I'm a CS student at Columbia who builds music tools and generative systems — not someone with a robotics background. I wanted something to show that I'd taken the concepts seriously, not just read about them.

But I don't think strategic and genuine are opposites. I built the thing. I debugged it for hours. I now understand REINFORCE well enough to explain exactly what was wrong with my first two implementations. That's real, even if the starting motivation was an application.

Whether robotics is what I want to do long-term — I'm still figuring that out. But I'm glad I spent a day here.

---

*Maria Showalter · Columbia CS · [mariashowalter.com](https://mariashowalter.com) · [github.com/Mariaareadne1](https://github.com/Mariaareadne1)*
