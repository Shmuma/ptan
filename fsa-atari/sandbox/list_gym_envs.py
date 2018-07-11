import gym
import gym_fsa_atari

all_envs = gym.envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
fsa_ids = [x for x in env_ids if 'fsa' in x]
print(fsa_ids)