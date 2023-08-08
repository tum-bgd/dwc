import numpy
import torch


class HDF5Dataset(torch.utils.data.Dataset):
    '''
    a minimal implementation for hdf5 in PyTorch
    '''
    def __init__(self, file, tag='tr'):
        super(HDF5Dataset, self).__init__()
        self.file = file
        self.tag = tag

    def __getitem__(self, index):
        image = torch.from_numpy(self.file[self.tag+'Image'][index]/255).float()
        label = torch.tensor(self.file[self.tag+'Label'][index])
        return (image, label)

    def __len__(self):
        return len(self.file[self.tag+'Label'])
