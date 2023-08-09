import argparse
import torch

# path
DATA_DIR_PREFIX = '../tiles'
DATASET_DIR = './drone.h5'
MODEL_WEIGHT_PATH = './trained_weights.pt'
PRUNED_MODEL_WEIGHT_PATH = './pruned_trained_weights.pt'
QUANT_CONFIG_PATH = './quant_config.json'

# label
LABEL2NUM_MAP = {
    'debris': 0,
    'forest': 1,
    'water' : 2,
    'other' : 3
}

NUM2LABEL_MAP = {
    0: 'debris',
    1: 'forest',
    2: 'water',
    3: 'other'
}

# image
N_CHANNEL = 3
TILE_W = 64
TILE_H = 64

# train & test
TR_RATIO = 0.8
TR_BATCHSIZE = 64
TE_BATCHSIZE = 1000
LOG_INTERVAL = 20

NO_CUDA = False
USE_CUDA = not NO_CUDA and torch.cuda.is_available()

DEVICE = torch.device("cpu")
if USE_CUDA:
    DEVICE = torch.device("cuda")

N_EPOCH = 20

# on-board
BOARD = "DPUCZDX8G_ISA1_B4096"
BOARD_INPUT = torch.randn([1, N_CHANNEL, TILE_H, TILE_W], dtype=torch.float32)    # NCHW
