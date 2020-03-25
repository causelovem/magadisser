import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as f
import random
import os
# import biotite
# import biotite.structure as struc
# import biotite.structure.io as strucio
# import biotite.application.dssp as dssp
import torch_geometric
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
from gaeModel import AutoEncoder
import config as cfg
import pdb2Gdata as p2d


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class readData(torch_geometric.data.Dataset):
    def __init__(self, fileDir, files):
        self.fileDir = fileDir
        self.files = files
        self.threshold = 7
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if cfg.pdbFile:
            # p2d.pdb2Gdata(os.path.join(self.fileDir, self.files[index]))
            return p2d.pdb2Gdata(self.fileDir, self.files[index])
        else:
            return torch.load(os.path.join(self.fileDir, self.files[index]))


set_seed(23)

fileDir = cfg.fileDir
dataList = os.listdir(fileDir)
dataList = dataList[:50]

validateLength = int(len(dataList) * cfg.validatePart)
dataSizes = [len(dataList) - validateLength, validateLength]
dataTrainRaw, dataValidateRaw = torch.utils.data.random_split(dataList, dataSizes)

dataTrain = readData(fileDir, dataTrainRaw)
dataValidate = readData(fileDir, dataValidateRaw)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = cfg.device
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

trainLoader = torch_geometric.data.DataLoader(dataTrain, batch_size=cfg.batchSize,
                                              num_workers=cfg.numWorkers, shuffle=True)
validateLoader = torch_geometric.data.DataLoader(dataValidate, batch_size=cfg.batchSize,
                                                 num_workers=cfg.numWorkers, shuffle=True)

model = AutoEncoder()
# model = model.double()
lossType = nn.MSELoss()
# lossType = f.nll_loss
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

model = model.to(device)

for epoch in range(cfg.epochsNum):
    print('Epoch {}/{}:'.format(epoch, cfg.epochsNum - 1), flush=True)

    model.train()
    for data in trainLoader:
        # print(data)

        data = data.to(device)
        preds = model(data).to(device)

        loss = lossType(preds, data.x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.4f}'.format(float(loss)), flush=True)

    model.eval()
    for data in validateLoader:
        data = data.to(device)
        preds = model(data).to(device)

        loss = lossType(preds, data.x)
    print('Validation Loss: {:.4f}'.format(float(loss)), flush=True)
