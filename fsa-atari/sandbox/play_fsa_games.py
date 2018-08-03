import gym
import gym_fsa_atari

import pygame
import sys
import time
import matplotlib
try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass

import pyglet.window as pw

from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))


def play(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None):
    """Allows one to play the game using keyboard.
    To simply play the game use:
        play(gym.make("Pong-v3"))
    Above code works also if env is wrapped, so it's particularly useful in
    verifying that the frame-level preprocessing does not render the game
    unplayable.
    If you wish to plot real time statistics as you play, you can use
    gym.utils.play.PlayPlot. Here's a sample code for plotting the reward
    for last 5 second of gameplay.
        def callback(obs_t, obs_tp1, rew, done, info):
            return [rew,]
        env_plotter = EnvPlotter(callback, 30 * 5, ["reward"])
        env = gym.make("Pong-v3")
        play(env, callback=env_plotter.callback)
    Arguments
    ---------
    env: gym.Env
        Environment to use for playing.
    transpose: bool
        If True the output of observation is transposed.
        Defaults to true.
    fps: int
        Maximum number of steps of the environment to execute every second.
        Defaults to 30.
    zoom: float
        Make screen edge this many times bigger
    callback: lambda or None
        Callback if a callback is provided it will be executed after
        every step. It takes the following input:
            obs_t: observation before performing action
            obs_tp1: observation after performing action
            action: action that was executed
            rew: reward that was received
            done: whether the environemnt is done or not
            info: debug info
    keys_to_action: dict: tuple(int) -> int or None
        Mapping from keys pressed to action performed.
        For example if pressed 'w' and space at the same time is supposed
        to trigger action number 2 then key_to_action dict would look like this:
            {
                # ...
                sorted(ord('w'), ord(' ')) -> 2
                # ...
            }
        If None, default key_to_action mapping for that env is used, if provided.
    """

    obs_s = env.observation_space.spaces['image']
    assert type(obs_s) == gym.spaces.box.Box
    assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

    if transpose:
        video_size = env.observation_space.spaces['image'].shape[1], env.observation_space.spaces['image'].shape[0]
    else:
        video_size = env.observation_space.spaces['image'].shape[0], env.observation_space.spaces['image'].shape[1]

    if zoom is not None:
        video_size = int(video_size[0] * zoom), int(video_size[1] * zoom)

    pressed_keys = []
    running = True
    env_done = True

    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()


    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            try:
                action = keys_to_action[tuple(sorted(pressed_keys))]
            except KeyError:
                action = 0 # NOOP
            prev_obs = obs
            obs, rew, env_done, info = env.step(action)
            # print(info)
            if callback is not None:
                callback(prev_obs, obs['image'], action, rew, env_done, info)
        if obs is not None:
            im = obs['image']
            logic = obs['logic']
            ram = env.env._get_ram()
            # didnt work: < 80
            # print("{:3d} {:3d} {:3d} {:3d}".format(ram[78], ram[76], ram[83], ram[84]))
            # print(ram[85:90])
            print(logic)
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = im.repeat(3, axis=2)
            display_arr(screen, im, transpose=transpose, video_size=video_size)

        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if event.key in relevant_keys:
                    pressed_keys.append(event.key)
                elif event.key == 27:
                    running = False
            elif event.type == pygame.KEYUP:
                if event.key in relevant_keys:
                    pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                video_size = event.size
                screen = pygame.display.set_mode(video_size)
                print(video_size)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data     = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]))
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)


if __name__ == '__main__':
    env = gym.make("fsa-SpaceInvadersNoFrameskip-v4")
    play(env, zoom=4, fps=40)