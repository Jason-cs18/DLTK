import os
import torch
from torch import optim, nn
from torchvision.models import resnet18, resnet50
from torchvision.transforms import Resize, Grayscale
from timm import create_model  # Use timm for pre-trained Vision
import lightning as L
from torch.nn import functional as F
# import torch_tensorrt

# Define resnet18 model with PyTorch
class TorchResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load ResNet18 and modify the input/output layers for MNIST
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust for MNIST classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Define the LightningModule for ResNet18
class LitResNet50(L.LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load ResNet50 and modify the input/output layers for MNIST
        self.model = resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust for MNIST classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx=0) -> torch.Tensor:
        x, y = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Define the LightningModule for ResNet18
class LitResNet18(L.LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load ResNet18 and modify the input/output layers for MNIST
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single-channel input
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust for MNIST classes

    # use torch.compile(self.model) to accelerate the model
    # def configure_model(self):
    #     if self.model is not None:
    #         return
        
        # self.model = torch.compile(
        #     model,
        #     backend="torch_tensorrt",
        #     options={
        #         "inputs": [torch_tensorrt.Input((1, 3, 224, 224))],  # Specify input shape
        #         "enabled_precisions": {torch.float32},  # Use FP32 precision (or FP16 for faster inference)
        #     }
        # )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx=0) -> torch.Tensor:
        x, y = batch
        return self(x)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

# Define the LightningModule for Vision Transformer
class LitVisionTransformer(L.LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Load a pre-trained Vision Transformer from timm
        self.model = create_model("vit_base_patch16_224", pretrained=True)

        # Modify the first convolutional layer to accept 1 input channel
        original_conv = self.model.patch_embed.proj
        self.model.patch_embed.proj = nn.Conv2d(1, original_conv.out_channels,
                                                kernel_size=original_conv.kernel_size,
                                                stride=original_conv.stride,
                                                padding=original_conv.padding,
                                                bias=False) # Keep bias consistent

        # Initialize the new weights (you might want to experiment with different initializations)
        with torch.no_grad():
            self.model.patch_embed.proj.weight[:, 0, :, :] = original_conv.weight.sum(dim=1)

        self.model.head = nn.Linear(self.model.head.in_features, num_classes)  # Adjust for MNIST classes
        # Define model-specific preprocessing (you might not need Grayscale here if the input is already grayscale)
        self.resize = Resize((224, 224))
        # self.grayscale_to_rgb = Grayscale(num_output_channels=3) # Consider removing this

    # use torch.compile(self.model) to accelerate the model
    # def configure_model(self):
    #     if self.model is not None:
    #         return
        
    #     self.model = torch.compile(self.model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply model-specific preprocessing
        x = self.resize(x)
        # x = self.grayscale_to_rgb(x) # Consider removing this
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx=0) -> torch.Tensor:
        x, _ = batch  # Only return predictions
        return self(x)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer