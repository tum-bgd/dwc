DATA_DIR_PREFIX = '../tiles'
DATASET_DIR = './drone.h5'

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

N_CHANNEL = 3
TILE_W = 64
TILE_H = 64