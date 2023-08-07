import h5py
import torch

from _config import *
from _model import *


with h5py.File(DATASET_DIR, 'r') as hf:
    image = hf['image']
    label = hf['label']

    print(torch.from_numpy(image[[1,4,6,7]]).size())
