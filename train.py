import os
import sys
from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import cv2
import mlflow
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import KVWriter, Logger, HumanOutputFormat

from utils.check_gpu import check_gpu
from flappy_bird_env import FlappyBirdGym


SAVE_DIR = "flappy_bird_training_log"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Hardware Check: Auto-detect Apple Metal (MPS)
check_gpu()


class ResizeAndGrayscaleWrapper(gym.ObservationWrapper):
    """
    Downsamples the image to 84x84 and converts to grayscale.
    This drastically reduces the input size for the CNN.
    """

    def __init__(self, env):
        super().__init__(env)
        # New shape: (84, 84, 1) -> Height, Width, Channel
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # 1. Convert to Grayscale (OpenCV expects H, W, C)
        # Note: Your env returns (Height, Width, Channels) already, which is good.
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # 2. Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # 3. Add the channel dimension back: (84, 84) -> (84, 84, 1)
        return np.expand_dims(resized, axis=-1)


class MLflowOutputFormat(KVWriter):
    """
    A custom translator that takes Stable Baselines3 training metrics
    and pipes them directly into MLflow's numeric logging.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # Only log numerical data (ignore strings or excluded keys)
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step=step)


class MLflowTrainingCallback(BaseCallback):
    def __init__(
        self, save_freq: int, save_dir: str, log_freq: int = 1000, verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.log_freq = log_freq
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                mlflow.log_metric(
                    "episode_reward", float(episode["r"]), step=self.num_timesteps
                )
                mlflow.log_metric(
                    "episode_length", float(episode["l"]), step=self.num_timesteps
                )
                if "score" in info:
                    mlflow.log_metric(
                        "episode_score", float(info["score"]), step=self.num_timesteps
                    )

        if self.n_calls % self.log_freq == 0:
            logger_values = getattr(self.model.logger, "name_to_value", {})
            metrics_to_track = {
                "train_loss": "train/loss",
                "train_n_updates": "train/n_updates",
                "rollout_exploration_rate": "rollout/exploration_rate",
                "rollout_ep_rew_mean": "rollout/ep_rew_mean",
                "rollout_ep_len_mean": "rollout/ep_len_mean",
                "time_fps": "time/fps",
            }
            for metric_name, logger_key in metrics_to_track.items():
                value = logger_values.get(logger_key)
                if value is not None and np.isscalar(value):
                    mlflow.log_metric(
                        metric_name, float(value), step=self.num_timesteps
                    )

        if self.n_calls % self.save_freq == 0:
            checkpoint_name = f"flappy_model_{self.num_timesteps}_steps"
            checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
            self.model.save(checkpoint_path)
            mlflow.log_artifact(
                f"{checkpoint_path}.zip", artifact_path="intermediate_models"
            )

        return True


def main():
    mlflow.set_tracking_uri("http://34.32.83.247:5000")
    mlflow.set_experiment("FlappyBird_DQN")
    train_params = {
        "buffer_size": 100000,  # Memory of past experiences
        "learning_starts": 10000,  # Wait 10k steps before training (fill buffer)
        "batch_size": 32,
        "exploration_fraction": 0.2,  # Explore 20% of the time, then purely exploit
        "exploration_final_eps": 0.02,  # Never stop exploring completely (2% random)
        "target_update_interval": 1000,  # Update the "Target Network" every 1k steps
        "train_freq": 4,  # Train every 4 steps
        "gradient_steps": 4,  # Train 4 times per update (learn faster)
        "learning_rate": 3e-4,  # Made the learning rate 3x more
        "skip_frames": 1,
        "checkpoint_freq": 100000,
    }

    with mlflow.start_run(run_name="dqn-high-lr-v2"):
        mlflow.log_params(train_params)
        # Create directories for saving models and logs
        models_dir = "models/DQN_FlappyBird"
        log_dir = "logs"
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # --- 2. Environment Setup ---
        # Create the base environment
        env = FlappyBirdGym()

        # Wrap it to monitor performance (logs reward per episode to Tensorboard)
        env = Monitor(env, log_dir)

        # Apply our custom Resize/Grayscale wrapper
        env = ResizeAndGrayscaleWrapper(env)

        # Vectorize it (SB3 requires VecEnv for FrameStack)
        env = DummyVecEnv([lambda: env])

        # Stack 4 frames (Dimensions become 84x84x4)
        # This allows the AI to see velocity (up vs down)
        env = VecFrameStack(env, n_stack=4, channels_order="last")

        # --- 3. The Model (DQN) ---
        model = DQN(
            "CnnPolicy",
            env,
            buffer_size=train_params["buffer_size"],
            learning_starts=train_params["learning_starts"],
            batch_size=train_params["batch_size"],
            exploration_fraction=train_params["exploration_fraction"],
            exploration_final_eps=train_params["exploration_final_eps"],
            target_update_interval=train_params["target_update_interval"],
            train_freq=train_params["train_freq"],
            gradient_steps=train_params["gradient_steps"],
            learning_rate=train_params["learning_rate"],
            verbose=1,
            tensorboard_log="logs",
        )

        # --- 4. The Logger ---
        custom_logger = Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
        )

        model.set_logger(custom_logger)
        # --- 5. Training + Artifact Callback ---
        # Log metrics and save/log checkpoints every 100,000 steps
        training_callback = MLflowTrainingCallback(
            save_freq=train_params["checkpoint_freq"],
            save_dir=models_dir,
            log_freq=1000,
        )

        # --- 6. Train ---
        print("Starting training...")
        # 1,000,000 steps is a good starting point for Flappy Bird
        model.learn(total_timesteps=1000000, callback=training_callback)

        # Save the final model
        model.save(f"{models_dir}/flappy_model_final")
        mlflow.pytorch.log_model(
            model.policy, "model", registered_model_name="FlappyBird_Brain"
        )
        print("Training Complete!")


if __name__ == "__main__":
    main()
