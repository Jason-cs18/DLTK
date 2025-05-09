"""
Test PyTorch Profile
"""
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import rich.console as console

console = console.Console()

if __name__ == "__main__":
    console.print("PyTorch Profile Test (ResNet18) on CPU", style="bold red")
    model = models.resnet18()
    model.eval()
    inputs = torch.randn(5, 3, 224, 224)
    
    # ProfilerActivity.CPU, ProfilerActivity.CUDA
    with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
        model(inputs)
    
    # self_cpu_time_total, self_cuda_time_total, self_cpu_memory_usage
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    
    console.print("PyTorch Profile Test (ResNet18) on GPU", style="bold red")
    model = models.resnet18().to("cuda:0")
    inputs = torch.randn(5, 3, 224, 224).to("cuda:0")
    
    # ProfilerActivity.CPU, ProfilerActivity.CUDA
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
        model(inputs)
    
    # self_cpu_time_total, self_cuda_time_total, self_cpu_memory_usage
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))