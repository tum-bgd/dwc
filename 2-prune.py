from pytorch_nndct import get_pruning_runner

from _config import *
from _data import *
from _model import *


parser = argparse.ArgumentParser(description='debris example - model pruning')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='saving the current model')
args = parser.parse_args()

# load saved model
model = Model(preTrained=True).to(DEVICE)

# prune a model (optimization)
with h5py.File(DATASET_DIR, 'r') as hf:
    trData = HDF5Dataset(hf, 'tr')
    teData = HDF5Dataset(hf, 'te')

    trDataLoader = torch.utils.data.DataLoader(trData, batch_size=TR_BATCHSIZE, shuffle=True, num_workers=0)
    teDataLoader = torch.utils.data.DataLoader(teData, batch_size=TE_BATCHSIZE, shuffle=True, num_workers=0)

    pruner = get_pruning_runner(model, BOARD_INPUT, 'iterative')
    pruner.ana(TeModel, args=(DEVICE, trDataLoader, True))
    
    prunedModel = pruner.prune(removal_ratio=0.1)

    TeModel(prunedModel, DEVICE, teDataLoader)
    if args.save_model:
        prunedModel.save_weights(PRUNED_MODEL_WEIGHT_PATH)
