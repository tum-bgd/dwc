from _config import *
from _data import *
from _model import *


parser = argparse.ArgumentParser(description='debris example - model training')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='saving the current model')
args = parser.parse_args()

with h5py.File(DATASET_DIR, 'r') as hf:
    trData = HDF5Dataset(hf, 'tr')
    teData = HDF5Dataset(hf, 'te')

    trDataLoader = torch.utils.data.DataLoader(trData, batch_size=TR_BATCHSIZE, shuffle=True, num_workers=0)
    teDataLoader = torch.utils.data.DataLoader(teData, batch_size=TE_BATCHSIZE, shuffle=True, num_workers=0)

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(1, N_EPOCH + 1):
        TrModel(model, DEVICE, trDataLoader, optimizer, epoch)
        TeModel(model, DEVICE, teDataLoader)
        scheduler.step()

    if args.save_model:
        model.save_weights(MODEL_WEIGHT_PATH)
