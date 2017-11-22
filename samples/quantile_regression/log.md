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
