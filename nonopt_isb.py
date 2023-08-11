import time

from _config import *
from _data import *
from _model import *


DEVICE = torch.device('cpu')  # cpu or gpu

model = Model(preTrained=True).to(DEVICE)
with h5py.File(DATASET_DIR, 'r') as hf:
    teData = HDF5Dataset(hf, 'te')
    teDataLoader = torch.utils.data.DataLoader(teData, batch_size=TE_BATCHSIZE, shuffle=True, num_workers=0)

    time1 = time.time()
    TeModel(model, DEVICE, teDataLoader)
    time2 = time.time()

    timetotal = time2 - time1
    runTotal = len(teData)
    fps = float(runTotal / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps, runTotal, timetotal))
    print(" ")
