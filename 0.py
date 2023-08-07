import os
import h5py
import torch

from torchvision.io import read_image

from _config import *


nImage = 0
for labelName in os.listdir(DATA_DIR_PREFIX):
    nImage += len(os.listdir(os.path.join(DATA_DIR_PREFIX, labelName)))

image = torch.empty((nImage, N_CHANNEL, TILE_W, TILE_H), dtype=torch.uint8)
label = torch.empty((nImage, ), dtype=torch.long)

i = 0
for labelName in LABEL2NUM_MAP.keys():
    for tileName in os.listdir(os.path.join(DATA_DIR_PREFIX, labelName)):
        image[i] = read_image(os.path.join(DATA_DIR_PREFIX, labelName, tileName))
        label[i] = LABEL2NUM_MAP[labelName]
        i += 1
        if i % 1000 == 0:
            print("{}/{}".format(i, nImage))
label = torch.nn.functional.one_hot(label, num_classes=4)

print(image.size())
print(label.size())

with h5py.File(DATASET_DIR, 'w') as hf:
    hf.create_dataset('image', data=image.numpy())
    hf.create_dataset('label', data=label.numpy())
