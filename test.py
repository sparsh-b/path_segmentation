import glob
import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import argparse
from torch.utils.data import DataLoader
import time
import cv2

from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from model import PathModel

parser = argparse.ArgumentParser()
parser.add_argument('--v', type=int, help='training run in lightning logs folder')
args = parser.parse_args()

mode = "test"
root = '..'
batch_size = 1
test_dataset = SimpleOxfordPetDataset(root, mode) #JMP_test
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

PATH = glob.glob('lightning_logs/version_{}/checkpoints/*ckpt'.format(args.v))
assert len(PATH) == 1
model = PathModel.load_from_checkpoint(PATH[0], arch="Unet", encoder_name="resnet18", in_channels=3, out_classes=1, mode=mode)
model.cuda()

out_path = 'predictions/test'
if not os.path.exists(out_path):
    os.makedirs(out_path)


for step, batch in enumerate(test_dataloader):
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask, filename in zip(batch["image"], batch["mask"], pr_masks, batch['filename']):
        cv2.imshow('prediction', pr_mask.cpu().numpy().squeeze())
        cv2.waitKey(1)

    # for image, gt_mask, pr_mask, filename in zip(batch["image"], batch["mask"], pr_masks, batch['filename']):
    #     plt.figure(figsize=(10, 5))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    #     plt.title("Image")
    #     plt.axis("off")

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    #     plt.title("Ground truth")
    #     plt.axis("off")

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(pr_mask.cpu().numpy().squeeze()) # just squeeze classes dim, because we have only one class
    #     plt.title("Prediction")
    #     plt.axis("off")

    #     #plt.show()
    #     plt.savefig('{}/pr{}'.format(out_path, filename))
    #     plt.close()
    fps = batch_size / (time.time() - start_time)
    print('fps:', fps)