import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader

from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from model import PathModel

root = '..'
train_dataset = SimpleOxfordPetDataset(root, "train")
test_dataset = SimpleOxfordPetDataset(root, "test")
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=n_cpu)

model = PathModel("Unet", "resnet18", in_channels=3, out_classes=1)

trainer = pl.Trainer(gpus=1, max_epochs=5000)
trainer.fit(model, train_dataloaders=train_dataloader)

#model.eval()
