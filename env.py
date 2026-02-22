from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from flappy_bird_env import FlappyBirdGym
from utils.resize_and_grayscale import ResizeAndGrayscaleWrapper

env = FlappyBirdGym(player="human")

env = ResizeAndGrayscaleWrapper(env)  # type: ignore
env = DummyVecEnv([lambda: env])  # type: ignore
env = VecFrameStack(env, n_stack=4, channels_order="last")  # type: ignore
