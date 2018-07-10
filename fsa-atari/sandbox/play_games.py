import gym
from gym.utils.play import play

env = gym.make("SpaceInvadersNoFrameskip-v4")

play(env, zoom=4)