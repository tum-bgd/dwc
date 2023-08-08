import os
import h5py
import random

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

print(image.size())
print(label.size())

index = list(range(0, nImage))
random.shuffle(index)

with h5py.File(DATASET_DIR, 'w') as hf:
    hf.create_dataset('trImage', data=image[index[:int(nImage*TR_RATIO)]].numpy())
    hf.create_dataset('teImage', data=image[index[int(nImage*TR_RATIO):]].numpy())
    hf.create_dataset('trLabel', data=label[index[:int(nImage*TR_RATIO)]].numpy())
    hf.create_dataset('teLabel', data=label[index[int(nImage*TR_RATIO):]].numpy())
