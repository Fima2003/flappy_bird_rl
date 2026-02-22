from flappy_bird_env import FlappyBirdGym
from stable_baselines3.common.env_checker import check_env

env = FlappyBirdGym()
# If this prints nothing and finishes, your environment is perfect.
check_env(env)
print("Environment check passed!")