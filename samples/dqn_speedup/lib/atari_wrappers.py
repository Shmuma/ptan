import numpy as np
from collections import deque
import gym
import gym_fsa_atari
from gym import spaces
import cv2

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, fsa=False):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)

        self.fsa = fsa
        # most recent raw observations (for max pooling across time steps)
        if fsa:
            self._obs_buffer = [None, None]
        else:
            self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype='uint8')

        self._skip       = skip

    def _step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if self.fsa:
            max_frame = np.concatenate((self._obs_buffer[0]['image'][None, ...],
                                        self._obs_buffer[1]['image'][None, ...]), axis=0).max(axis=0)
            # return max_frame, total_reward, done, info
            return {'image': max_frame, 'logic': self._obs_buffer[1]['logic']}, total_reward, done, info
        else:
            max_frame = self._obs_buffer.max(axis=0)
            return max_frame, total_reward, done, info

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, fsa=False):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.fsa = fsa
        if self.fsa:
            image_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
            logic_space = env.observation_space.spaces['logic']
            self.observation_space = spaces.Dict({'image': image_space, 'logic': logic_space})
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def _observation(self, frame):
        if self.fsa:
            logic = frame['logic']
            frame = frame['image']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.fsa:
            return {'image': frame[:, :, None], 'logic': logic}
        else:
            return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, fsa=False):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.fsa = fsa
        if fsa:
            shp = env.observation_space.spaces['image'].shape
            image_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)
            logic_space = spaces.MultiDiscrete([env.observation_space.spaces['logic'].n] * k)
            self.observation_space = spaces.Dict({'image': image_space, 'logic': logic_space})
        else:
            shp = env.observation_space.shape
            self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), fsa=self.fsa)

class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames, fsa=False):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        if fsa:
            self._frames = [x['image'] for x in frames]
            self._logics = [x['logic'] for x in frames]
        else:
            self._frames = frames

        self.fsa = fsa

    def __array__(self, dtype=None):
        if self.fsa:
            image_out = np.concatenate(self._frames, axis=2)
            # logic_out = np.vstack(self._logics)
            if dtype is not None:
                image_out = image_out.astype(dtype)
            out = image_out
            # out = {'image': image_out, 'logic': logic_out}
        else:
            out = np.concatenate(self._frames, axis=2)
            if dtype is not None:
                out = out.astype(dtype)
        return out

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env, fsa=False):
        super(ImageToPyTorch, self).__init__(env)
        self.fsa = fsa
        if fsa:
            old_shape = self.observation_space.spaces['image'].shape
            image_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0],
                                                                              old_shape[1]), dtype=np.uint8)
            self.observation_space = gym.spaces.Dict({'image': image_space,
                                                      'logic': env.observation_space.spaces['logic']})
        else:
            old_shape = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0],
                                                                          old_shape[1]), dtype=np.uint8)

    def _observation(self, observation):
        if self.fsa:
            return {'image': np.swapaxes(observation, 2, 0),
                    'logic': np.vstack(observation._logics)}
        else:
            return np.swapaxes(observation, 2, 0)

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False,
                    scale=False, pytorch_img=False, fsa=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, fsa=fsa)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4, fsa=fsa)
    if pytorch_img:
        env = ImageToPyTorch(env, fsa=fsa)
    return env

def make_atari(env_id, fsa=False):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4, fsa=fsa)
    return env
