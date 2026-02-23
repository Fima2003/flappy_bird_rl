import time
import numpy as np
from utils.fetch_model import model
from env import env
from utils.results_tracker import record_game

obs = env.reset()

# FPS Limiter setup
FPS = 20
current_steps = 0

while True:
    start_time = time.time()

    # Transpose obs from (1, 84, 84, 4) to (1, 4, 84, 84) for PyTorch
    obs_transposed = np.transpose(obs, (0, 3, 1, 2))

    # mlflow.pyfunc.predict returns just the action, not a tuple
    action = model.predict(obs_transposed)

    # Ensure action is a 1D numpy array
    action = np.array(action).flatten()

    obs, rewards, done, info = env.step(action)
    current_steps += 1
    env.render()

    if done:
        score = info[0].get("score", 0) if isinstance(
            info, (list, tuple)) else info.get("score", 0)
        record_game("python", score)
        obs = env.reset()

    # Cap the frame rate so you can actually watch it
    process_time = time.time() - start_time
    sleep_time = (1.0 / FPS) - process_time
    if sleep_time > 0:
        time.sleep(sleep_time)
