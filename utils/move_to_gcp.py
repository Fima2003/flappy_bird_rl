import mlflow
import mlflow.pytorch
from stable_baselines3 import DQN

# 1. Setup Cloud Connection
# Replace with your actual Google Cloud Run URL
CLOUD_MLFLOW_URL = "http://34.32.83.247:5000"
mlflow.set_tracking_uri(CLOUD_MLFLOW_URL)
mlflow.set_experiment("FlappyBird_DQN_CNN")

# 2. Define the parameters you used locally
train_params = {
    "buffer_size": 100000,
    "learning_starts": 10000,  # Wait 10k steps before training (fill buffer)
    "batch_size": 32,
    "exploration_fraction": 0.2,  # Explore 20% of the time, then purely exploit
    "exploration_final_eps": 0.02,  # Never stop exploring completely (2% random)
    "target_update_interval": 1000,  # Update the "Target Network" every 1k steps
    "train_freq": 4,  # Train every 4 steps
    "gradient_steps": 4,  # Train 4 times per update (learn faster)
    "learning_rate": 3e-4,  # Made the learning rate 3x more
}


def migrate_to_cloud():
    # 3. Load your local 'Brain'
    local_model_path = "models/DQN_FlappyBird/flappy_model_final.zip"
    print(f"Loading local model from {local_model_path}...")
    model = DQN.load(local_model_path)

    # 4. Push to Google Cloud
    with mlflow.start_run(run_name="manual-migration-v1"):
        print("Logging parameters to Cloud SQL...")
        mlflow.log_params(train_params)

        print("Uploading model to GCS and Registering Brain...")
        # We log the policy (the weights) and register it
        mlflow.pytorch.log_model(
            pytorch_model=model.policy,
            artifact_path="model",
            registered_model_name="FlappyBird_Brain",
        )

    print("✅ Migration Complete! Your bird is now in the Cloud Hub.")


if __name__ == "__main__":
    migrate_to_cloud()
