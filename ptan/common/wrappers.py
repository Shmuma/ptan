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


class FrameBuffer(ObservationWrapper):
    def __init__(self, env, n_frames=4):
        """A gym wrapper that returns last n_frames observations as a single observation.
        Useful for games like Atari and Doom with screen as input."""
        super(FrameBuffer, self).__init__(env)
        shape = (n_frames,) + env.observation_space.shape[1:]
        self.observation_space = Box(0.0, 1.0, shape)
        self.framebuffer = np.zeros(shape=(n_frames,) + env.observation_space.shape)

    def _reset(self):
        self.framebuffer = np.zeros_like(self.framebuffer)
        return super(FrameBuffer, self)._reset()

    def _observation(self, observation):
        self.framebuffer = np.concatenate((observation[None], self.framebuffer[:-1]))

        s = self.framebuffer.shape
        return np.reshape(self.framebuffer, newshape=(s[0]*s[1], s[2], s[3]))

