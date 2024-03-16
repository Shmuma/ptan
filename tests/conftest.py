import pytest
import gymnasium as gym


@pytest.fixture
def car_env() -> gym.Env:
    env = gym.make("MountainCar-v0")
    yield env
    env.close()


@pytest.fixture
def cartpole_env() -> gym.Env:
    env = gym.make("CartPole-v1")
    yield env
    env.close()
