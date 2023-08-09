from pytorch_nndct.apis import Inspector, torch_quantizer

from _config import *
from _data import *
from _model import *


parser = argparse.ArgumentParser(description='debris example - model evaluation and export')
parser.add_argument('--deploy', action='store_true', default=True,
                    help='generate model deploy related files')
args = parser.parse_args()

# load saved pruned model
prunedModel = Model(preTrained=True, pruned=True).to(DEVICE)

# include_bias_corr set to False because bias_corr.pth stores a dict with all values are None.
quantizer = torch_quantizer('test', prunedModel, BOARD_INPUT, device=DEVICE, quant_config_file=QUANT_CONFIG_PATH, target=BOARD)
quantModel = quantizer.quant_model
quantizer.load_ft_param()
with h5py.File(DATASET_DIR, 'r') as hf:
    teData = HDF5Dataset(hf, 'te')
    teDataLoader = torch.utils.data.DataLoader(teData, batch_size=1, shuffle=True, num_workers=0)
    TeModel(quantModel, DEVICE, teDataLoader, False)
    if args.deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()



