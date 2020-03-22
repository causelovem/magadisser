import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import random
import os
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Interpolate(nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        # x = interpolate(x, mode=self.mode, scale_factor=self.scale_factor)
        # return x
        return interpolate(x, mode=self.mode, scale_factor=self.scale_factor)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            # Interpolate(mode='bilinear', scale_factor=2),
            # nn.ConvTranspose2d(32, 32, kernel_size=3),
            # nn.ReLU(True),

            # Interpolate(mode='bilinear', scale_factor=2),
            # nn.ConvTranspose2d(32, 32, kernel_size=3),
            # nn.ReLU(True),

            # Interpolate(mode='bilinear', scale_factor=2),
            # nn.ConvTranspose2d(32, 32, kernel_size=3),
            # nn.ReLU(True),

            Interpolate(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 32, kernel_size=3),
            nn.ReLU(True)

            # Interpolate(mode='bilinear', scale_factor=2),
            # nn.ConvTranspose2d(32, 1, kernel_size=3)
            # nn.ReLU(True),

            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x


class readData(torch.utils.data.Dataset):
    def __init__(self, fileDir, files):
        self.fileDir = fileDir
        self.files = files
        self.threshold = 7
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        print(os.path.join(self.fileDir, self.files[index]))
        array = strucio.load_structure(os.path.join(self.fileDir, self.files[index]))
        if type(array) == biotite.structure.AtomArrayStack:
            array = array[0]
        # print(os.path.join(self.fileDir, self.files[index]))
        # print(type(array))

        ca = array[array.atom_name == "CA"]
        cell_list = struc.CellList(ca, cell_size=self.threshold)

        # cell_list = struc.CellList(array, cell_size=self.threshold)
        adj_matrix = cell_list.create_adjacency_matrix(self.threshold).astype(int)

        shape = adj_matrix.shape

        if shape[0] % 2 != 0:
            print(shape)
            adj_matrix = np.append(adj_matrix, np.zeros((1, shape[0]), dtype=float), axis=0)
            adj_matrix = np.append(adj_matrix, np.zeros((shape[0] + 1, 1), dtype=float), axis=1)
            print(adj_matrix.shape)

        # return torch.tensor(adj_matrix.astype('float'))
        return adj_matrix.astype('double')


set_seed(23)

fileDir = '/mnt/ssd1/prog/pdbFiles'
dataList = os.listdir(fileDir)
validatePart = 0.3
batchSize = 1
epochsNum = 5
numWorkers = 1

validateLength = int(len(dataList) * validatePart)
dataSizes = [len(dataList) - validateLength, validateLength]
dataTrainRaw, dataValidateRaw = torch.utils.data.random_split(dataList, dataSizes)

dataTrain = readData(fileDir, dataTrainRaw)
dataValidate = readData(fileDir, dataValidateRaw)

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
trainLoader = torch.utils.data.DataLoader(dataTrain, batch_size=batchSize,
                                          num_workers=numWorkers, shuffle=True)
validateLoader = torch.utils.data.DataLoader(dataValidate, batch_size=batchSize,
                                             num_workers=numWorkers, shuffle=True)

model = AutoEncoder()
model = model.double()
lossType = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

model = model.to(device)

for epoch in range(epochsNum):
    print('Epoch {}/{}:'.format(epoch, epochsNum - 1), flush=True)

    model.train()
    for data in trainLoader:
        print(data.shape)

        data = torch.stack((data,)).to(device)
        preds = model(data)

        print(preds.shape)

        loss = lossType(preds, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.4f}'.format(float(loss)), flush=True)

    model.eval()
    for data in validateLoader:
        data = torch.stack((data,)).to(device)
        preds = model(data)
        loss = lossType(preds, data)
    print('Validation Loss: {:.4f}'.format(float(loss)), flush=True)
