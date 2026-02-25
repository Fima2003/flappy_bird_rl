from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from flappy_bird import FlappyBird


class FlappyBirdGym(gym.Env):
    def __init__(self, player="AI"):
        super().__init__()

        self.action_space = Discrete(2)  # 0 - Do Nothing, 1 - Jump
        self.observation_space = Box(
            low=0, high=255, shape=(512, 288, 3), dtype=np.uint8
        )

        self.game = FlappyBird(player=player)

        self.max_steps = 2000
        self.current_step = 0

    def get_state(self):
        # Get the pixels
        pixels = self.game.get_screen_pixels()
        # switch from (h,w,c) -> (w,h,c)
        self.state = np.transpose(pixels, axes=(1, 0, 2)).astype(np.uint8)
        return self.state

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        self.game.reset()
        self.game.render()
        observation = self.get_state()
        self.current_step = 0

        info: dict[str, Any] = {}

        return observation, info

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False

        # FRAME SKIP: Repeat the action 4 times
        for i in range(4):
            self.current_step += 1

            # If the AI wants to jump (action==1), we only press the button
            # on the VERY FIRST frame (i==0).
            # On frames 1, 2, and 3, we force 'False' so gravity can work.
            if i == 0 and action == 1:
                flap_action = True
                total_reward -= 0.05  # Penalty for flapping (energy cost)
            else:
                flap_action = False

            # Step the physics
            done, passed_pipe = self.game.step(flap_action)

            # Render internal state
            self.game.render()

            # Accumulate Reward
            if done:
                terminated = True
                total_reward = -100  # High penalty for dying
                break

            if passed_pipe:
                total_reward += 10

            total_reward += 0.1

            if self.current_step >= self.max_steps:
                truncated = True
                break

        observation = self.get_state()
        info = {"score": self.game.score}

        return observation, total_reward, terminated, truncated, info
