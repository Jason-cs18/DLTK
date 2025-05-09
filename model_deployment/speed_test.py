"""
speed_test.py
compare the speed of the PyTorch model with and without torch.compile + torch_tensorrt
"""
import time

import torch
import torchvision
# import torch_tensorrt
import tensorrt
import rich.console as console

import onnxruntime as ort
from onnxconverter_common import float16

from models import TorchResNet18, LitResNet18, LitResNet50

def profile_model_latency(model, batch_sizes, device="cuda"):
    """
    Profiles the latency of a Lightning model (LitResNet18) for different batch sizes.

    Args:
        model (LitResNet18): The Lightning model to profile.
        batch_sizes (list): List of batch sizes to test.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        dict: A dictionary with batch sizes as keys and average latency (ms) as values.
    """
    # Move the model to the specified device
    model = model.cuda()
    model.eval()

    # Dictionary to store latency results
    latency_results = {}

    # Loop through each batch size
    for batch_size in batch_sizes:
        # Generate random input tensor
        inputs = torch.randn(batch_size, 1, 28, 28).cuda()  # Adjusted for MNIST input size

        # Warm-up runs (to stabilize GPU performance)
        for _ in range(5):
            _ = model(inputs)

        # Measure latency
        start_time = time.time()
        for _ in range(10):  # Run multiple iterations for averaging
            _ = model(inputs)
        end_time = time.time()

        # Calculate average latency in milliseconds
        avg_latency = (end_time - start_time) / 10 * 1000
        latency_results[batch_size] = avg_latency

        print(f"Batch size: {batch_size}, Average Latency: {avg_latency:.2f} ms")

    return latency_results


def profile_onnx_model_latency(onnx_model_path="resnet18.onnx", batch_sizes=[1], device="cuda"):
    """
    Profiles the latency of an ONNX model for different batch sizes.

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        batch_sizes (list): List of batch sizes to test.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        dict: A dictionary with batch sizes as keys and average latency (ms) as values.
    """
    # Load the ONNX model with ONNX Runtime
    # providers = ['CUDAExecutionProvider'] if device == "cuda" else ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 0 # 0:Verbose, 1:Info, 2:Warning, 3:Error
    # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # session_options.optimized_model_filepath = onnx_model_path
    # Provider options (can be empty if defaults are fine, but good to be explicit)
    # cuda_provider_options = {'device_id': '0'} 
    # providers = [("CUDAExecutionProvider", cuda_provider_options)]
    if device == "cuda":
        # providers = [
        #     'TensorrtExecutionProvider',
        #     ('CUDAExecutionProvider', {
        #     'device_id': 0,
        #     'arena_extend_strategy': 'kNextPowerOfTwo',
        #     'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        #     'cudnn_conv_algo_search': 'EXHAUSTIVE',
        #     'do_copy_in_default_stream': True,
        #     }),
        #     'CPUExecutionProvider',
        # ]
        # providers = [
        #     ('TensorrtExecutionProvider', {
        #         'device_id': 0,
        #         'trt_fp16_enable': False,  # CRITICAL: Disable FP16 for TensorRT EP
        #         # You can add other TRT EP options here if needed
        #         # 'trt_int8_enable': False, # Ensure INT8 is also off
        #         # 'trt_engine_cache_enable': True,
        #         # 'trt_engine_cache_path': './ort_trt_fp32_cache', 
        #         # 'trt_max_workspace_size': 2147483648 
        #     }),
        #     ('CUDAExecutionProvider', {
        #         'device_id': 0,
        #         # CUDA EP generally respects model precision, but no harm being explicit if options existed
        #         # 'precision_mode': 'fp32' # Example syntax if such an option existed (it usually doesn't directly)
        #     })
            # CPU EP is usually not needed as a fallback if GPU EPs are specified and available
        # ]
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        # providers = [
        #     ('TensorrtExecutionProvider', {
        #         'device_id': 0,                       # Select GPU to execute
        #         'trt_max_workspace_size': 2147483648, # Set GPU memory usage limit
        #         'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
        #     }),
        #     ('CUDAExecutionProvider', {
        #         'device_id': 0,
        #         'arena_extend_strategy': 'kNextPowerOfTwo',
        #         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        #         'cudnn_conv_algo_search': 'EXHAUSTIVE',
        #         'do_copy_in_default_stream': True,
        #     })
        # ]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers, session_options=session_options)
    print(session.get_providers())
    # print("For small models, data transfer time is bottleneck and CPU mode is faster")
    
    # Get input name for the ONNX model
    input_name = session.get_inputs()[0].name

    # Dictionary to store latency results
    latency_results = {}

    # Loop through each batch size
    for batch_size in batch_sizes:
        # Generate random input tensor
        inputs = torch.randn(batch_size, 1, 28, 28).numpy()  # Adjusted for MNIST input size

        # Warm-up runs (to stabilize GPU performance)
        for _ in range(5):
            session.run(None, {input_name: inputs})

        # Measure latency
        start_time = time.time()
        for _ in range(10):  # Run multiple iterations for averaging
            session.run(None, {input_name: inputs})
        end_time = time.time()

        # Calculate average latency in milliseconds
        avg_latency = (end_time - start_time) / 10 * 1000
        latency_results[batch_size] = avg_latency

        print(f"Batch size: {batch_size}, Average Latency (ONNX): {avg_latency:.2f} ms")

    return latency_results


def profile_onnx_model_latency_with_gpu_input_binding(
    onnx_model_path="resnet18.onnx",
    batch_sizes=[1],
    device="cuda",
    num_warmup_runs=20,
    num_timed_runs=100
):
    """
    Profiles the latency of an ONNX model for different batch sizes,
    using IOBinding for GPU inputs if device is "cuda".

    Args:
        onnx_model_path (str): Path to the ONNX model file.
        batch_sizes (list): List of batch sizes to test.
        device (str): Device to run the model on ("cuda" or "cpu").
        num_warmup_runs (int): Number of warm-up inferences.
        num_timed_runs (int): Number of inferences to average for latency.

    Returns:
        dict: A dictionary with batch sizes as keys and average latency (ms) as values.
    """
    session_options = ort.SessionOptions()
    # You can set graph optimization level if needed, e.g.:
    # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3 # 0:Verbose, 1:Info, 2:Warning, 3:Error (3=Error for less noise)

    if device == "cuda":
        if not CUPY_AVAILABLE:
            print("Error: CuPy is required for GPU input binding but not installed. Aborting CUDA test.")
            return {bs: float('nan') for bs in batch_sizes}
        # Order: TensorRT > CUDA > CPU. TensorRT EP will use IOBinding if available.
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                # "trt_engine_cache_enable": True, # Recommended for repeated runs
                # "trt_engine_cache_path": "./trt_cache",
                # "trt_fp16_enable": True, # If your model supports FP16
            }),
            ("CUDAExecutionProvider", {"device_id": 0, "gpu_mem_limit": 2 * 1024 * 1024 * 1024}),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(onnx_model_path, providers=providers, sess_options=session_options)
    except Exception as e:
        print(f"Error creating ONNX Runtime session for device {device} with providers {providers}: {e}")
        return {bs: float('nan') for bs in batch_sizes}
        
    print(f"Using ONNX Runtime device: {device.upper()}")
    print(f"Active ONNX Runtime providers: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    # Assuming model's expected input dtype is float32, common for image models
    # If your model uses a different input type, adjust np.float32 and cp.float32 accordingly.
    input_dtype = np.float32

    latency_results = {}
    io_binding = None # Initialize outside the loop if structure is consistent

    if device == "cuda" and 'CUDAExecutionProvider' in session.get_providers() or \
       device == "cuda" and 'TensorrtExecutionProvider' in session.get_providers():
        io_binding = session.io_binding() # Create IOBinding object

    for batch_size in batch_sizes:
        # Define the input shape. User's example was (batch, 1, 28, 28)
        current_input_shape = (batch_size, 1, 28, 28)
        print(f"\nProfiling for batch size: {batch_size} with shape {current_input_shape}")

        if device == "cuda" and io_binding:
            # 1. Create input data directly on GPU using CuPy
            try:
                inputs_gpu = cp.random.randn(*current_input_shape).astype(input_dtype)
                
                # 2. Bind the GPU input buffer
                io_binding.bind_input(
                    name=input_name,
                    device_type='cuda',
                    device_id=0,  # Assuming GPU device ID 0
                    element_type=input_dtype,
                    shape=inputs_gpu.shape,
                    buffer_ptr=inputs_gpu.data.ptr # Get the GPU data pointer
                )

                # (Optional) Bind output buffer to GPU if you want to keep output on GPU
                # output_name = session.get_outputs()[0].name
                # output_metadata = session.get_outputs()[0]
                # output_shape_template = output_metadata.shape # e.g. [None, num_classes]
                # actual_output_shape = [batch_size if dim is None or isinstance(dim, str) else dim for dim in output_shape_template]
                # output_gpu = cp.empty(actual_output_shape, dtype=input_dtype)
                # io_binding.bind_output(
                #     name=output_name,
                #     device_type='cuda',
                #     device_id=0,
                #     element_type=input_dtype,
                #     shape=tuple(actual_output_shape),
                #     buffer_ptr=output_gpu.data.ptr
                # )
            except Exception as e:
                print(f"  Error during CuPy input creation or binding for batch size {batch_size}: {e}")
                latency_results[batch_size] = float('nan')
                continue

            # Warm-up runs
            # print(f"  Warm-up runs ({num_warmup_runs})...")
            for _ in range(num_warmup_runs):
                session.run_with_iobinding(io_binding) # Pass the binding object

            # Measure latency
            # print(f"  Timed runs ({num_timed_runs})...")
            start_time = time.perf_counter()
            for _ in range(num_timed_runs):
                session.run_with_iobinding(io_binding)
            end_time = time.perf_counter()

            # Retrieve outputs if needed (they will be OrtValues on GPU if output was bound to GPU)
            # outputs_ortvalue_gpu = io_binding.get_outputs()
            # If you need them on CPU: outputs_numpy = [ov.numpy() for ov in outputs_ortvalue_gpu]

        else:  # CPU Execution Path
            # Generate random input tensor on CPU
            inputs_cpu = torch.randn(*current_input_shape).numpy().astype(input_dtype)

            # Warm-up runs
            # print(f"  Warm-up runs ({num_warmup_runs})...")
            for _ in range(num_warmup_runs):
                session.run(None, {input_name: inputs_cpu})

            # Measure latency
            # print(f"  Timed runs ({num_timed_runs})...")
            start_time = time.perf_counter()
            for _ in range(num_timed_runs):
                session.run(None, {input_name: inputs_cpu})
            end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) * 1000 / num_timed_runs
        latency_results[batch_size] = avg_latency_ms

        print(f"  Batch size: {batch_size}, Average Latency: {avg_latency_ms:.3f} ms")

    return latency_results


if __name__ == "__main__":
    # try:
    #     import cupy as cp
    #     CUPY_AVAILABLE = True
    # except ImportError:
    #     CUPY_AVAILABLE = False
    #     print("Warning: CuPy is not installed. GPU input binding will not be available. Install with 'pip install cupy-cudaXX'.")

    console = console.Console()
    console.log("Checking environments...")
    console.log("torch version:", torch.__version__)
    console.log("CUDA available:", torch.cuda.is_available())
    console.log("torchvision version:", torchvision.__version__)
    # console.log("torch_tensorrt version:", torch_tensorrt.__version__)
    # console.log("tensorrt version:", tensorrt.__version__)
    console.log("onnxruntime version:", ort.__version__)
    console.log("GPU available for onnx:", ort.get_device())
    console.log("Model: ResNet50")
    console.print("Running speed test (pytorch lightning & gpu)...", style="bold red")
    # ckpt_path = "./logs/resnet/lightning_logs/version_0/checkpoints/epoch=9-step=8600.ckpt"
    
    # PyTorch native
    # model = TorchResNet18()
    # checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(checkpoint["state_dict"])
    # model.eval()
    # batch_sizes = [1, 8, 16, 32, 64]
    # latency_results = profile_model_latency(model, batch_sizes, device="cuda")
    
    model = LitResNet50(num_classes=10)
    # model = LitResNet50.load_from_checkpoint(ckpt_path)
    # model = model.float()
    model.eval()
    batch_sizes = [1, 8, 16, 32, 64, 128]
    latency_results_pytorch_gpu = profile_model_latency(model, batch_sizes, device="cuda")
    console.print("Running speed test (microsoft onnx & gpu)...", style="bold red")
    # model_onnx_dir = "resnet18.onnx"
    # model.to_onnx(model_onnx_dir, export_params=True, input_sample=torch.randn(1, 1, 28, 28))
    # export onnx model (gpu)
    model = model.cuda()
    model_onnx_path = "resnet18_dynamic_batch.onnx" # Changed name for clarity
    export_params = True
    input_sample = torch.randn(1, 1, 28, 28).float().cuda() # NCHW format
    
    input_names = ["input_image"]
    output_names = ["output_predictions"] # Adjust if your model has multiple outputs
    dynamic_axes_config = {
        input_names[0]: {0: 'batch_size'},
        output_names[0]: {0: 'batch_size'}
    }
    model.to_onnx(
        model_onnx_path,
        input_sample=input_sample,
        export_params=export_params,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_config,
        # verbose=True
        # opset_version=12  # Recommended: Use a recent stable opset_version (e.g., 11, 12, 13 ... 17)
    )
    console.print("Exported ONNX model to:", model_onnx_path)
    # batch_sizes = [1, 8, 16, 32]
    latency_results_onnx_gpu = profile_onnx_model_latency(onnx_model_path=model_onnx_path, batch_sizes=batch_sizes, device="cuda")
    # console.print("Running speed test (onnxruntime on gpu with fp16)...", style="bold red")
    # import onnx
    # model = onnx.load(model_onnx_path)
    # model_fp16 = float16.convert_float_to_float16(model)
    # onnx.save(model_fp16, "onnx_fp16.onnx")
    # latency_results_onnx = profile_onnx_model_latency(onnx_model_path="onnx_fp16.onnx", batch_sizes=batch_sizes, device="cuda")
    console.print("Running speed test (microsoft onnx & cpu)...", style="bold red")
    latency_results_onnx_cpu = profile_onnx_model_latency(onnx_model_path=model_onnx_path, batch_sizes=batch_sizes, device="cpu")
    from rich.table import Table
    table = Table(show_header=True, header_style="bold magenta", title="Latency Results for ResNet50 (ms/batch)", show_lines=True)
    table.add_column("Batch Size", justify="center")
    table.add_column("PyTorch GPU", justify="center")
    table.add_column("ONNX CPU", justify="center")
    table.add_column("ONNX GPU", justify="center")
    for batch_size in batch_sizes:
        table.add_row(
            str(batch_size),
            f"{latency_results_pytorch_gpu.get(batch_size, 'N/A'):.2f}",
            f"{latency_results_onnx_cpu.get(batch_size, 'N/A'):.2f}",
            f"{latency_results_onnx_gpu.get(batch_size, 'N/A'):.2f}"
        )
    console.print(table)
    # console.log("Latency results (torch-tensorrt) TBD...")
    # console.log("Running speed test (torch-tensorrt)...")
    # model_tensorrt = TorchResNet18()
    # checkpoint = torch.load(ckpt_path)
    # model_tensorrt.load_state_dict(checkpoint["state_dict"])
    # model_tensorrt = torch.compile(
    #         model_tensorrt,
    #         backend="torch_tensorrt",
    #         options={
    #             "inputs": [torch_tensorrt.Input((1, 3, 224, 224))],  # Specify input shape
    #             "enabled_precisions": {torch.float32},  # Use FP32 precision (or FP16 for faster inference)
    #         }
    #     )
    # model_tensorrt.eval()
    # import torch._dynamo
    # # torch._dynamo.config.suppress_errors = True
    # batch_sizes = [1]
    # latency_results = profile_model_latency(model_tensorrt, batch_sizes, device="cuda:0")
    
    
    
