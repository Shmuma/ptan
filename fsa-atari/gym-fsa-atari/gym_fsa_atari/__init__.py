from gym.envs.registration import register

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    # space_invaders should yield SpaceInvaders-v0
    name = ''.join([g.capitalize() for g in game.split('_')])

    nondeterministic = False

    register(
        id='fsa-{}-v0'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game, 'repeat_action_probability': 0.25},
        max_episode_steps=10000,
        nondeterministic=nondeterministic,
    )

    register(
        id='fsa-{}-v4'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    # Standard Deterministic (as in the original DeepMind paper)
    if game == 'space_invaders':
        frameskip = 3
    else:
        frameskip = 4

    # Use a deterministic frame skip.
    register(
        id='fsa-{}Deterministic-v0'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    register(
        id='fsa-{}Deterministic-v4'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game, 'frameskip': frameskip},
        max_episode_steps=100000,
        nondeterministic=nondeterministic,
    )

    register(
        id='fsa-{}NoFrameskip-v0'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
        max_episode_steps=frameskip * 100000,
        nondeterministic=nondeterministic,
    )

    # No frameskip. (Atari has no entropy source, so these are
    # deterministic environments.)
    register(
        id='fsa-{}NoFrameskip-v4'.format(name),
        entry_point='gym_fsa_atari.envs:FsaAtariEnv',
        kwargs={'game': game, 'frameskip': 1}, # A frameskip of 1 means we get every frame
        max_episode_steps=frameskip * 100000,
        nondeterministic=nondeterministic,
    )