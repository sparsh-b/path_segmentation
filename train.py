import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import argparse

from pprint import pprint
from torch.utils.data import DataLoader
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from model import PathModel

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, help='0 or 1 or 2 for the current 3-fold cross-validation')
args = parser.parse_args()

root = '..'
train_dataset = SimpleOxfordPetDataset(root, "train_fold{}".format(args.fold))
test_dataset = SimpleOxfordPetDataset(root, "test_fold{}".format(args.fold))
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
