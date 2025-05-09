import ray
from ray import serve
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict

@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
class ResNetTRTDeployment:
    def __init__(self, engine_path, input_name, output_names, input_shape, dtype):
        self.engine_path = engine_path
        self.input_name = input_name
        self.output_names = output_names
        self.input_shape = input_shape
        self.dtype = dtype
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        self.stream = cuda.Stream()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        for binding in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(binding)
            tensor = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = np.prod(tensor)
            gpu_buffer = cuda.mem_alloc(size * dtype().itemsize)
            bindings.append(int(gpu_buffer))
            if self.engine.binding_is_input(binding):
                inputs.append({'name': name, 'shape': tensor, 'dtype': dtype, 'device': gpu_buffer})
            else:
                host_buffer = np.empty(tensor, dtype=dtype)
                outputs.append({'name': name, 'shape': tensor, 'dtype': dtype, 'device': gpu_buffer, 'host': host_buffer})
        return inputs, outputs, bindings

    def preprocess(self, image_bytes: bytes):
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = image.reshape(self.input_shape)
        image = image.astype(self.dtype) / 255.0
        image = np.ascontiguousarray(image)
        return image

    def postprocess(self, output):
        return np.argmax(output)

    async def predict(self, image_file: UploadFile) -> Dict:
        try:
            image_bytes = await image_file.read()
            processed_input = self.preprocess(image_bytes)
            cuda.memcpy_htod_async(self.inputs[0]['device'], processed_input, self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

            for output in self.outputs:
                cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            self.stream.synchronize()

            output_data = self.outputs[0]['host']
            result = self.postprocess(output_data)

            # DEBUG: Confirm type
            print("Output type:", type(result), "| Value:", result)

            return {"prediction": int(result)}
        except Exception as e:
            print("Error during inference:", str(e))
            return JSONResponse(content={"error": str(e)}, status_code=500)

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

# Configuration parameters
engine_file_path = "/codebase/model_development/resnet50_engine_fp16.trt"
input_tensor_name = "input_image"
output_tensor_names = ["output_predictions"]
input_shape = (1, 1, 28, 28)
input_dtype = np.float32

@serve.deployment
@serve.ingress(app)
class ModelDeploymentWrapper:
    def __init__(self, engine_path, input_name, output_names, input_shape, dtype):
        # Initialize ResNetTRTDeployment with the provided arguments
        self.model = ResNetTRTDeployment(
            engine_path=engine_path,
            input_name=input_name,
            output_names=output_names,
            input_shape=input_shape,
            dtype=dtype
        )

    @app.post("/predict")
    async def predict_endpoint(self, image: UploadFile = File(...)) -> Dict:
        return await self.model.predict(image)

# 2: Deploy the deployment.
if __name__ == "__main__":
    ray.init()
    serve.run(ModelDeploymentWrapper.bind(
        engine_file_path=engine_file_path,  # Pass arguments to bind
        input_name=input_tensor_name,
        output_names=output_tensor_names,
        input_shape=input_shape,
        dtype=input_dtype
    ), route_prefix="/")
    print("Ray Serve deployment started. Send POST requests to http://localhost:8000/predict with an image file.")
