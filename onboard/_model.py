import os
import torch.nn.functional as F
import warnings

from _config import *


class Model(torch.nn.Module):
    def __init__(self, preTrained=False, pruned=False):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1),  # in_channel, out_channel, kernel_size, stride
            torch.nn.MaxPool2d(2),         # kernel_size
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 32, 3, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(1152, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(NUM2LABEL_MAP.keys())),
            torch.nn.LogSoftmax(dim=1)
        )
        if preTrained:
            if pruned:
                self.load_weights(PRUNED_MODEL_WEIGHT_PATH)
                warnings.warn("model with trained & pruned weights")
            else:
                self.load_weights(MODEL_WEIGHT_PATH)
                warnings.warn("model with trained weights")
        else:
            warnings.warn("model with random weights")

    def forward(self, x):
        return self.model(x)

    def load_weights(self, dir):
        if os.path.exists(dir):
            return self.load_state_dict(torch.load(dir, map_location=DEVICE))
        else:
            warnings.warn("No model weight file found. Return model with random weights.")

    def save_weights(self, dir):
        if os.path.exists(dir):
            os.remove(dir)
            warnings.warn("Model weight file exists and will be removed. New weights will be saved.")
        torch.save(self.state_dict(), dir)


def TrModel(model, device, train_loader, optimizer, epoch, dry_run=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def TeModel(model, device, test_loader, returnAcc=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if returnAcc:
        return correct / len(test_loader.dataset)


if __name__ == '__main__':
    model = Model()
    a = model(torch.randn(22, 3, 64, 64))
    print(a.size())