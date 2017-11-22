# Part 1: DQN speed up

Starting with basic DQN version with target net and classical training regime 
(sample rainbow/01_dqn_basic.py).

Goal: make it as fast as possible on GTX 1080Ti and PyTorch 0.2.0

Basic speed: 159 f/s in the beginning of the training (epsilon=1.0)

## Game play on CPU

First attempt: move game play on CPU to reduce GPU data copy and single batch processing.
Result: 120 f/s

## Steps batch

Perform multiple steps and increase batch size. As replay buffer is large, it shouldn't 
make much difference if we perform several steps (not too many) between optimisation runs.

* Steps=1: 159 f/s, GPU 39%
* Steps=2: 202 f/s, GPU 37%
* Steps=4: 232 f/s, GPU 35%: convergence slowed down
* Steps=8: 246 f/s, GPU 32%
* Steps=16: 254 f/s, GPU 30%: doesn't converge

## Play in a separate process

By using torch.multiprocessing implemented play inside the separate process, which produces experience 
via shared queue.

Results: 
* Steps=1: 221 f/s
* Steps=2: 315 f/s
* Steps=3: 370 f/s
* Steps=4: 393 f/s: convergence slowed down, so steps=2 is optimal

Next: move gameplay on CPU. Upd: speed is much worse

## CPU consumption in training and play process

Training process uses 100% CPU core, which is weird, as it only sampling and preparing data for GPU.
Need to experiment where CPU goes.

Baseline is Steps=3 version (370 f/s), during buffer fill: 550 f/s

1. Get rid of sampling (train on constant buffer): 376 f/s
2. No optimiser.step(): 400 f/s
3. No calc_loss_dqn: 552 f/s (same as buffer fill)

## Async in cuda() calls

Increased from 370 f/s to 384 f/s

## Latest atari wrappers

Baseline is Steps=3, 384 f/s

* New version with the same set of wrappers as before: 445 f/s (GPU 74%)
* Without episodic_life: 450 f/s
* Without clip_rewards: 445 f/s
* Without both episodic_life and clip_rewards: 440 f/s
* Without pytorch reshape: 390 f/s

## Final numbers

159 f/s -> 445 f/s: **180% speedup**

## Bonus: put play on a different GPU

Frames between play and train nets sync:
* 1: 430 f/s
* 4: 455 f/s
* 10: 457 f/s

So, it doesn't worth it
