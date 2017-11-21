# Part 1: DQN speed up

Starting with basic DQN version with target net and classical training regime 
(sample rainbow/01_dqn_basic.py).

Goal: make it as fast as possible on GTX 1080Ti and PyTorch 0.2.0

Basic speed: 159 f/s in the beginning of the training (epsilon=1.0)

First attempt: move game play on CPU to reduce GPU data copy and single batch processing.
Result: 120 f/s

