import numpy as np
import tritonclient.http as httpclient

# 1) Create the Triton HTTP client
client = httpclient.InferenceServerClient(url="localhost:8000")

# 2) Prepare random FP32 input matching your model’s config
data = np.random.randn(1, 1, 28, 28).astype(np.float16)

# 3) Create InferInput and set data
infer_input = httpclient.InferInput("input_image", data.shape, "FP16")
infer_input.set_data_from_numpy(data)  # <-- This sets the 'data' field

# 4) Request output
infer_output = httpclient.InferRequestedOutput("output_predictions")

# 5) Perform inference
response = client.infer(model_name="resnet50",
                        inputs=[infer_input],
                        outputs=[infer_output])
result = response.as_numpy("output_predictions")
print(result)
