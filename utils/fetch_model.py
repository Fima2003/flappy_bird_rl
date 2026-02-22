import mlflow
import mlflow.pyfunc

# Point to your Cloud VM
mlflow.set_tracking_uri("http://34.32.83.247:5000")

# Load the "Production" version of your model
model_name = "FlappyBird_Brain"
model_alias = "production"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
