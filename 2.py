import h5py
from pytorch_nndct import get_pruning_runner

from _config import *
from _data import *
from _model import *


# load saved model
model = Model(MODEL_WEIGHT_PATH).to(DEVICE)

# prune a model (optimization)
with h5py.File(DATASET_DIR, 'r') as hf:
    trData = HDF5Dataset(hf, 'tr')
    teData = HDF5Dataset(hf, 'te')

    trDataLoader = torch.utils.data.DataLoader(trData, batch_size=TR_BATCHSIZE, shuffle=True, num_workers=0)
    teDataLoader = torch.utils.data.DataLoader(teData, batch_size=TE_BATCHSIZE, shuffle=True, num_workers=0)

    inputs = torch.randn([64, 1, 28, 28], dtype=torch.float32)
    pruner = get_pruning_runner(model, inputs, 'iterative')
    pruner.ana(TeModel, args=(DEVICE, trDataLoader, True))
    prunedModel = pruner.prune(removal_ratio=0.1)

    print(TeModel(prunedModel, DEVICE, teDataLoader, True))