import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Ensures CUDA context is initialized and cleaned up
import numpy as np
import time
import os
# import traceback # Rich console.print_exception handles this
from rich.console import Console
from rich.table import Table # For a nice summary table

# Initialize Rich Console
console = Console()

# Define a global TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) # TensorRT 8.5.1.7

def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    console.print(f"Loading TensorRT engine from: [cyan]{engine_file_path}[/cyan]...", style="bold blue")
    if not os.path.exists(engine_file_path):
        console.print(f"[bold red][ERROR] Engine file not found: {engine_file_path}[/bold red]")
        return None
    try:
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                console.print("[bold red][ERROR] Failed to deserialize the engine.[/bold red]")
                return None
            console.print("Engine loaded successfully.", style="green")
            console.print(f"  Engine has [blue]{engine.num_optimization_profiles}[/blue] optimization profile(s).")
            return engine
    except Exception as e:
        console.print(f"[bold red][ERROR] Failed to load engine: {e}[/bold red]")
        console.print_exception(show_locals=True)
        return None

def profile_trt_engine(engine: trt.ICudaEngine, batch_sizes: list, input_tensor_name: str, base_input_shape_chw: tuple,
                       num_warmup_runs: int = 20, num_timed_runs: int = 100):
    latency_results_ms = {}

    # Create ONE execution context to be reused across different batch sizes
    try:
        with engine.create_execution_context() as context: # Context created ONCE
            console.print("Global execution context created.")

            if engine.num_optimization_profiles > 0:
                try:
                    active_prof_idx = context.active_optimization_profile
                    console.print(f"  Context initial active optimization profile index: [yellow]{active_prof_idx}[/yellow]")
                    if active_prof_idx < 0: # Should be 0 for an engine with 1 profile
                        console.print(f"  [bold red]WARNING: Context reports initial active_optimization_profile < 0 ({active_prof_idx}). This might cause issues.[/bold red]")
                        # If the "mOptimizationProfile >= 0 failed" error returns with this setup,
                        # you *might* need to explicitly set profile 0 ONCE here for this persistent context:
                        # console.print("  Attempting to explicitly set optimization profile to 0 for the persistent context...")
                        # setup_stream = cuda.Stream()
                        # if context.set_optimization_profile_async(0, setup_stream.handle):
                        #     setup_stream.synchronize()
                        #     console.print(f"  Optimization profile explicitly set to 0. New active profile: {context.active_optimization_profile}")
                        # else:
                        #     console.print("  [bold red]Failed to explicitly set optimization profile 0 for persistent context.[/bold red]")
                        # del setup_stream
                except AttributeError:
                    console.print("  [yellow][WARN] context.active_optimization_profile attribute not found.[/yellow]")
                except Exception as e_prof:
                    console.print(f"  [yellow][WARN] Error querying context.active_optimization_profile: {e_prof}[/yellow]")

            for batch_size in batch_sizes: # Loop for batch sizes, REUSING the same context
                actual_input_shape = (batch_size,) + base_input_shape_chw
                console.rule(f"[bold cyan]Profiling Batch Size: {batch_size} | Input Shape: {actual_input_shape}[/bold cyan]")

                input_dtype_engine = engine.get_tensor_dtype(input_tensor_name)
                input_dtype_np = trt.nptype(input_dtype_engine)
                host_input_data = np.random.randn(*actual_input_shape).astype(input_dtype_np)

                host_outputs = {}
                device_inputs = {} 
                device_outputs = {}
                bindings_addrs = [0] * engine.num_bindings
                current_batch_device_allocations = [] # To free memory for this batch iteration

                try:
                    # Set input shape for the CURRENT BATCH on the PERSISTENT context
                    if not context.set_input_shape(input_tensor_name, host_input_data.shape):
                        console.print(f"  [bold red][ERROR] Failed to set input shape for '{input_tensor_name}' to {host_input_data.shape}.[/bold red]")
                        latency_results_ms[batch_size] = float('nan')
                        continue
                    # console.print(f"  Set input shape for '[magenta]{input_tensor_name}[/magenta]' to: [yellow]{host_input_data.shape}[/yellow]") # Verbose

                    for i in range(engine.num_bindings):
                        binding_name = engine.get_tensor_name(i)
                        tensor_mode = engine.get_tensor_mode(binding_name)
                        tensor_shape_dims = context.get_tensor_shape(binding_name) # Get shape from context
                        tensor_shape_tuple = tuple(tensor_shape_dims)
                        tensor_dtype_trt = engine.get_tensor_dtype(binding_name)
                        tensor_dtype_np = trt.nptype(tensor_dtype_trt)
                        
                        # Minor optimization for console output, can be commented out if too verbose
                        # temp_array_for_size = np.empty(tensor_shape_tuple, dtype=tensor_dtype_np)
                        # mem_size = temp_array_for_size.nbytes
                        # console.print(f"    Binding '[magenta]{binding_name}[/magenta]': Shape=[yellow]{tensor_shape_tuple}[/yellow], Dtype=[green]{tensor_dtype_np}[/green], Size=[blue]{mem_size}[/blue] bytes")


                        if tensor_mode == trt.TensorIOMode.INPUT:
                            if binding_name == input_tensor_name:
                                device_mem = cuda.mem_alloc(host_input_data.nbytes)
                                device_inputs[binding_name] = (device_mem, host_input_data)
                            else:
                                other_host_input = np.zeros(tensor_shape_tuple, dtype=tensor_dtype_np)
                                device_mem = cuda.mem_alloc(other_host_input.nbytes)
                                device_inputs[binding_name] = (device_mem, other_host_input)
                            current_batch_device_allocations.append(device_mem)
                            bindings_addrs[i] = int(device_mem)
                        else: 
                            host_buffer = cuda.pagelocked_empty(tensor_shape_tuple, dtype=tensor_dtype_np)
                            device_mem = cuda.mem_alloc(host_buffer.nbytes)
                            host_outputs[binding_name] = host_buffer
                            device_outputs[binding_name] = device_mem
                            current_batch_device_allocations.append(device_mem)
                            bindings_addrs[i] = int(device_mem)
                    
                    inference_stream = cuda.Stream()

                    for _ in range(num_warmup_runs):
                        for name, (d_mem, h_data) in device_inputs.items():
                            cuda.memcpy_htod_async(d_mem, np.ascontiguousarray(h_data), inference_stream)
                        context.execute_async_v2(bindings=bindings_addrs, stream_handle=inference_stream.handle)
                        for name, h_data_out_buffer in host_outputs.items():
                            cuda.memcpy_dtoh_async(h_data_out_buffer, device_outputs[name], inference_stream)
                        inference_stream.synchronize()

                    timings = []
                    for _ in range(num_timed_runs):
                        start_time = time.perf_counter()
                        for name, (d_mem, h_data) in device_inputs.items():
                            cuda.memcpy_htod_async(d_mem, np.ascontiguousarray(h_data), inference_stream)
                        context.execute_async_v2(bindings=bindings_addrs, stream_handle=inference_stream.handle)
                        for name, h_data_out_buffer in host_outputs.items():
                            cuda.memcpy_dtoh_async(h_data_out_buffer, device_outputs[name], inference_stream)
                        inference_stream.synchronize()
                        end_time = time.perf_counter()
                        timings.append(end_time - start_time)
                    
                    if num_timed_runs > 0 :
                        avg_latency_s = sum(timings) / num_timed_runs
                        avg_latency_ms = avg_latency_s * 1000
                        latency_results_ms[batch_size] = avg_latency_ms
                        console.print(f"  [bold green]Avg Latency BS {batch_size}: {avg_latency_ms:.3f} ms[/bold green]")
                    else:
                        latency_results_ms[batch_size] = float('nan')
                        console.print(f"  [yellow]No timed runs performed for BS {batch_size}.[/yellow]")


                except Exception as e_inner:
                    console.print(f"  [bold red][ERROR] Inner loop for BS {batch_size}: {e_inner}[/bold red]")
                    console.print_exception(show_locals=True)
                    latency_results_ms[batch_size] = float('nan')
                finally:
                    # Free device memory allocated for THIS BATCH ITERATION
                    for d_mem_ptr in current_batch_device_allocations:
                        if d_mem_ptr:
                            try:
                                d_mem_ptr.free()
                            except cuda.LogicError as le: # Catch if already freed or context issue
                                console.print(f"    [dim yellow]Note: PyCUDA LogicError freeing memory for BS {batch_size} (likely already freed): {le}[/dim yellow]")
                    if 'inference_stream' in locals() and inference_stream: del inference_stream
            
            console.print("Global execution context and its associated resources (like implicitly selected optimization profile state) destroyed as 'with' block exits.")

    except Exception as e_outer:
        console.print(f"[bold red][ERROR] Outer error related to context creation/usage: {e_outer}[/bold red]")
        console.print_exception(show_locals=True)
        for bs_fallback in batch_sizes: # Populate remaining batch sizes with NaN if error occurs early
            if bs_fallback not in latency_results_ms:
                latency_results_ms[bs_fallback] = float('nan')
    
    return latency_results_ms

# --- Main execution block (`if __name__ == "__main__":`) remains the same ---
if __name__ == "__main__":
    console.rule("[bold yellow]TensorRT Engine Profiling Script[/bold yellow]")
    # --- Configuration ---
    engine_file_path = "/codebase/model_development/resnet50_engine_fp16.trt"
    input_tensor_name = "input_image"
    base_input_shape_chw = (1, 28, 28)
    batch_sizes_to_test = [1, 8, 16, 32, 64, 128] 
    num_warmup = 50
    num_timed = 200
    # --- End Configuration ---

    console.print(f"TensorRT Version: [bold blue]{trt.__version__}[/bold blue]") # TRT 8.5.1.7
    console.print(f"Engine Path: [cyan]{engine_file_path}[/cyan]")
    console.print(f"Input Tensor Name: [magenta]{input_tensor_name}[/magenta]")
    console.print(f"Base Input Shape (CHW): [yellow]{base_input_shape_chw}[/yellow]")
    console.print(f"Batch Sizes to Test: [yellow]{batch_sizes_to_test}[/yellow]")
    console.print(f"Warm-up Runs: {num_warmup}, Timed Runs: {num_timed}")

    engine = load_engine(engine_file_path)

    if engine:
        # ... (validation logic for input_tensor_name remains the same) ...
        is_valid_input_name_and_type = False
        try:
            mode = engine.get_tensor_mode(input_tensor_name)
            if mode == trt.TensorIOMode.INPUT:
                is_valid_input_name_and_type = True
                console.print(f"\nEngine Input '[magenta]{input_tensor_name}[/magenta]' confirmed as INPUT. Expected Dtype: [green]{trt.nptype(engine.get_tensor_dtype(input_tensor_name))}[/green]")
            else:
                console.print(f"[bold red][ERROR] Tensor '{input_tensor_name}' is not an INPUT (it's an {str(mode).split('.')[-1]}).[/bold red]")
        except RuntimeError: 
            is_valid_input_name_and_type = False 
        except Exception as e:
            console.print(f"[bold red][ERROR] Unexpected error validating input tensor name '{input_tensor_name}': {e}[/bold red]")
            console.print_exception(show_locals=True)
            is_valid_input_name_and_type = False

        if not is_valid_input_name_and_type:
            console.print(f"[bold red][ERROR] Input tensor '{input_tensor_name}' not found or not an input. Available I/O tensors:[/bold red]")
            # ... (listing available tensors logic remains same) ...
            exit()

        latency_results = profile_trt_engine(
            engine,
            batch_sizes_to_test,
            input_tensor_name,
            base_input_shape_chw,
            num_warmup_runs=num_warmup,
            num_timed_runs=num_timed
        )

        console.rule("[bold yellow]Latency Measurement Summary[/bold yellow]")
        # ... (table printing logic remains same) ...
        if latency_results:
            summary_table = Table(show_header=True, header_style="bold magenta", title="Performance Results (ms/batch)")
            summary_table.add_column("Batch Size", style="dim", width=12, justify="center")
            summary_table.add_column("Avg Latency (ms)", justify="right")
            summary_table.add_column("Throughput (FPS)", justify="right")

            for bs, lat_ms in latency_results.items():
                if lat_ms is not float('nan') and lat_ms > 0:
                    throughput_fps = (bs / lat_ms) * 1000
                    summary_table.add_row(str(bs), f"{lat_ms:8.3f}", f"{throughput_fps:8.2f}")
                else:
                    summary_table.add_row(str(bs), "N/A (error)" if lat_ms is float('nan') else "N/A (zero/neg)", "N/A")
            console.print(summary_table)
        else:
            console.print("No latency results to display.")
    else:
        console.print("[bold red]Exiting due to engine loading failure.[/bold red]")

    # ---- ADD THIS SECTION for explicit cleanup ----
    if 'engine' in locals() and engine is not None:
        console.print("Explicitly deleting TensorRT engine object before script exit...", style="dim")
        del engine
        # Optionally, force a CUDA context sync here if PyCUDA is still active
        # This might not be necessary if pycuda.autoinit handles it, but can be tried.
        # try:
        #     cuda.Context.synchronize() # Requires an active context
        #     console.print("CUDA context synchronized after engine deletion.", style="dim")
        # except Exception as e_sync:
        #     console.print(f"[yellow]Note: Could not synchronize context after engine deletion: {e_sync}[/yellow]", style="dim")
    # ---- END ADDED SECTION ----

    console.print("\nProfiling script finished.", style="bold green")
    # pycuda.autoinit's atexit handler will run after this to clean up the CUDA context.