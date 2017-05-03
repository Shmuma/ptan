"""basic wrappers, useful for reinforcement learning on gym envs"""
# Stolen here: AgentNet/agentnet/experiments/openai_gym/wrappers.py
import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper,Wrapper
from gym.spaces.box import Box


class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        return img


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that returns last n_frames observations as a single observation.
        Useful for games like Atari and Doom with screen as input."""
        super(FrameBuffer, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [n_frames,]+list(env.observation_space.shape))
        self.framebuffer = np.zeros([n_frames,]+list(env.observation_space.shape))

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns 4-frame buffer"""
        new_obs, r, done, info = self.env.step(action)
        self.update_buffer(new_obs)
        return self.framebuffer, r, done, info

    def update_buffer(self, obs):
        """push new observation to the buffer, remove the earliest one"""
        self.framebuffer = np.vstack([obs[None], self.framebuffer[:-1]])
