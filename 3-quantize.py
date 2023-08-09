from pytorch_nndct.apis import Inspector, torch_quantizer

from _config import *
from _data import *
from _model import *


# load saved pruned model
prunedModel = Model(preTrained=True, pruned=True).to(DEVICE)

# inspect target device for best performances
inspector = Inspector(BOARD)  # by target name (kv260)
inspector.inspect(prunedModel, BOARD_INPUT, device=DEVICE, image_format="png")

# calibration
quantizer = torch_quantizer('calib', prunedModel, BOARD_INPUT, device=DEVICE, target=BOARD)
quantModel = quantizer.quant_model
with h5py.File(DATASET_DIR, 'r') as hf:
    trData = HDF5Dataset(hf, 'tr', loadSmallPortion=True)
    trDataLoader = torch.utils.data.DataLoader(trData, batch_size=1, shuffle=True, num_workers=0)
    quantizer.fast_finetune(TeModel, (quantModel, DEVICE, trDataLoader, True))
    quantizer.export_quant_config()
