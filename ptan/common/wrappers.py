import typing as tt
import gymnasium as gym
from gymnasium import spaces
import collections
import numpy as np
from stable_baselines3.common import atari_wrappers


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = gym.spaces.Box(
            low=obs.low.min(), high=obs.high.max(),
            shape=new_shape, dtype=obs.dtype)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0), dtype=obs.dtype)
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: tt.Optional[int] = None,
              options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(self.buffer)


def wrap_dqn(env: gym.Env, stack_frames: int = 4,
             episodic_life: bool = True, clip_reward: bool = True,
             noop_max: int = 0) -> gym.Env:
    """
    Apply a common set of wrappers for Atari games.
    :param env: Environment to wrap
    :param stack_frames: count of frames to stack, default=4
    :param episodic_life: convert life to end of episode
    :param clip_reward: reward clipping
    :param noop_max: how many NOOP actions to execute
    :return: wrapped environment
    """
    assert 'NoFrameskip' in env.spec.id
    env = atari_wrappers.AtariWrapper(
        env, clip_reward=clip_reward, noop_max=noop_max,
        terminal_on_life_loss=episodic_life)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, stack_frames)
    return env
