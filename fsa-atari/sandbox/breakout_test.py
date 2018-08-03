# Import the gym module
import gym
import gym_fsa_atari
import pdb

# Create a breakout environment
# env = gym.make('Breakout-ram-v0')
env = gym.make('fsa-Breakout-v0')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

pdb.set_trace()

is_done = False
while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    print(reward)
    # Render
    env.render()
env.close()