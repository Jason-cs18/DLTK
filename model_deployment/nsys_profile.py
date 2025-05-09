import torch
import torchvision.models as models
#from torch.profiler import profile, record_function, ProfilerActivity
import rich.console as console
console = console.Console()

if __name__ == "__main__":
    console.print("Nsight Systems Test (ResNet18) on GPU", style="bold red")
    torch.cuda.nvtx.range_push("model")
    model = models.resnet18().cuda()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("inputs")
    inputs = torch.randn(1, 3, 224, 224).cuda()
    torch.cuda.nvtx.range_pop()
    model.eval()

    torch.cuda.nvtx.range_push("forward")
    with torch.no_grad():
        for i in range(30):
            torch.cuda.nvtx.range_push(f"iteration {i}")
            model(inputs)
            torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()