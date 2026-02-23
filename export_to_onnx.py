import os
import torch
import mlflow.pytorch

# Point to your Cloud VM
mlflow.set_tracking_uri("http://34.32.83.247:5000")

# Load the "Production" version of your model
model_name = "FlappyBird_Brain"
model_alias = "production"

print(f"Fetching model {model_name}@{model_alias} from MLflow...")
model = mlflow.pytorch.load_model(f"models:/{model_name}@{model_alias}")
model.eval()

# Create dummy input: Batch Size 1, 4 Channels (stacked frames), 84x84
dummy_input = torch.zeros(1, 4, 84, 84, dtype=torch.float32)

output_path = "static/flappy_bird.onnx"
print(f"Exporting model to {output_path}...")

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("Export successful!")
