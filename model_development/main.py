import torch
from lightning.pytorch.cli import LightningCLI
from models import LitResNet18, LitVisionTransformer
from datasets import MNISTDataModule

def cli_main():
    cli = LightningCLI(datamodule_class=MNISTDataModule)

if __name__ == "__main__":
    cli_main()
    
    # job = "vit" # resnet, vit
    # if job == "resnet":
    #     # Initialize the data module
    #     data_module = datasets.MNISTDataModule(data_dir="./data")
        
    #     # Initialize the model
    #     model = models.LitResNet18(num_classes=10)
        
    #     # Initialize the trainer
    #     # L.trainer = models.Trainer(max_epochs=5)
    #     trainer = L.Trainer(max_epochs=10)
        
    #     # Train the model
    #     trainer.fit(model, data_module)
    #     # Test the model
    #     trainer.test(model, data_module)
    #     # Predict with the model
    #     # predictions = trainer.predict(model, data_module)
    # elif job == "vit":
    #     # Initialize the data module
    #     data_module = datasets.MNISTDataModule(data_dir="./data")
        
    #     # Initialize the model
    #     model = models.LitVisionTransformer(num_classes=10)
    #     # model = torch.compile(model) # accelerate the model
    #     # Initialize the trainer
    #     # L.trainer = models.Trainer(max_epochs=5)
    #     trainer = L.Trainer(limit_train_batches=10, max_epochs=3)
    #     # trainer = L.Trainer(max_epochs=10)
        
    #     # Train the model
    #     trainer.fit(model, data_module)
    #     # Test the model
    #     trainer.test(model, data_module)