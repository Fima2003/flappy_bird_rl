import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("static/flappy_bird.onnx")

# Get input and output details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape

print(f"Input: name={input_name}, shape={input_shape}, type={input_type}")
print(f"Output: name={output_name}, shape={output_shape}")

# Test with 255 and 1.0 logic
dummy_255 = np.random.randint(0, 255, size=(1, 4, 84, 84)).astype(np.float32)
dummy_1 = dummy_255 / 255.0

out_255 = session.run([output_name], {input_name: dummy_255})[0]
out_1 = session.run([output_name], {input_name: dummy_1})[0]

print("Out with 0-255 input:", out_255)
print("Out with 0-1 input:", out_1)
