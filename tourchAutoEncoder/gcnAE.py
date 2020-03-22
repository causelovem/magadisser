import numpy as np
import torch
import torch.nn as nn
# from torch.nn.functional import interpolate
import torch.nn.functional as f
import random
import os
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


elements = [
    'C',
    'N',
    'O',
    'S',
    'F',
    'Si',
    'P',
    'Cl',
    'Br',
    'Mg',
    'Na',
    'Ca',
    'Fe',
    'As',
    'Al',
    'I',
    'B',
    'V',
    'K',
    'Tl',
    'Yb',
    'Sb',
    'Sn',
    'Ag',
    'Pd',
    'Co',
    'Se',
    'Ti',
    'Zn',
    'H',
    'Li',
    'Ge',
    'Cu',
    'Au',
    'Ni',
    'Cd',
    'In',
    'Mn',
    'Zr',
    'Cr',
    'Pt',
    'Hg',
    'Pb',
    'Unknown'
]

elementsDict = dict([[elements[i], float(i)] for i in range(len(elements))])

residuales = [
    'ALA',
    'AMP',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'GOL',
    'HIS',
    'HOH',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'MPD',
    'PHE',
    'PRO',
    'SER',
    'SO4',
    'THR',
    'TRP',
    'TYR',
    'VAL',
    'Unknown'
]

residualesDict = dict([[residuales[i], float(i)] for i in range(len(residuales))])

atoms = [
    'N',
    'CA',
    'C',
    'O',
    'CB',
    'CG',
    'CD1',
    'CD2',
    'CE1',
    'CE2',
    'CZ',
    'OH',
    'CG1',
    'CG2',
    'CD',
    'CE',
    'NZ',
    'OE1',
    'NE2',
    'NE',
    'NH1',
    'NH2',
    'OG',
    'ND1',
    'OE2',
    'OD1',
    'OD2',
    'OG1',
    'ND2',
    'NE1',
    'CE3',
    'CZ2',
    'CZ3',
    'CH2',
    'SD',
    'OXT',
    'P',
    'O1P',
    'O2P',
    'O3P',
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    'N9',
    'C8',
    'N7',
    'C5',
    'C6',
    'N6',
    'N1',
    'C2',
    'N3',
    'C4',
    'Unknown'
]

atomsDict = dict([[atoms[i], float(i)] for i in range(len(atoms))])


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 32)

        self.unconv1 = GCNConv(32, 16)
        self.unconv2 = GCNConv(16, 8)

        # self.encoder = nn.Sequential(
        #     GCNConv(8, 16),
        #     GCNConv(16, 32)
        # )

        # self.decoder = nn.Sequential(
        #     GCNConv(32, 16),
        #     GCNConv(16, 8)
        # )

    # def forward(self, x):
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = f.relu(self.conv1(x, edge_index))
        x = f.relu(self.conv2(x, edge_index))
        x = f.relu(self.unconv1(x, edge_index))
        x = f.relu(self.unconv2(x, edge_index))
        # x = self.encoder(x)
        # print(x.shape)
        # x = self.decoder(x)
        # print(x.shape)
        return x


# class readData(torch.utils.data.Dataset):
class readData(torch_geometric.data.Dataset):
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

        # ca = array[array.atom_name == "CA"]
        # cell_list = struc.CellList(ca, cell_size=self.threshold)

        cell_list = struc.CellList(array, cell_size=self.threshold)
        # adj_matrix = cell_list.create_adjacency_matrix(self.threshold).astype(int)
        adj_matrix = cell_list.create_adjacency_matrix(self.threshold)

        # (adj_matrix[adj_matrix == True].shape[0] - 5385) / 2
        edge_index = [[], []]

        # arrayShape = array.shape[0]
        # for i in range(arrayShape - 1):
        #     for j in range(i + 1, arrayShape):
        #         if struc.distance(array[i], array[j]) <= self.threshold:
        #             edge_index[0].append(i)
        #             edge_index[1].append(j)

        nodeFeatures = []
        arrayShape = array.shape[0]
        # shape = adj_matrix.shape
        for i in range(arrayShape - 1):
            for j in range(i + 1, arrayShape):
                if adj_matrix[i][j]:
                    edge_index[0].append(i)
                    edge_index[1].append(j)

            nodeFeatures.append(
                list(array.coord[i]) +
                # [atomsDict[array.atom_name[i]]] +
                [atomsDict.get(array.atom_name[arrayShape - 1], atomsDict['Unknown'])] +
                # [elementsDict[array.element[i]]] +
                [elementsDict.get(array.element[arrayShape - 1], elementsDict['Unknown'])] +
                [array.res_id[i]] +
                # [residualesDict[array.res_name[i]]] +
                [residualesDict.get(array.res_name[arrayShape - 1], residualesDict['Unknown'])] +
                [float(array.hetero[i])]
            )
        nodeFeatures.append(
            list(array.coord[arrayShape - 1]) +
            # [atomsDict[array.atom_name[arrayShape - 1]]] +
            [atomsDict.get(array.atom_name[arrayShape - 1], atomsDict['Unknown'])] +
            # [elementsDict[array.element[arrayShape - 1]]] +
            [elementsDict.get(array.element[arrayShape - 1], elementsDict['Unknown'])] +
            [array.res_id[arrayShape - 1]] +
            # [residualesDict[array.res_name[arrayShape - 1]]] +
            [residualesDict.get(array.res_name[arrayShape - 1], residualesDict['Unknown'])] +
            [float(array.hetero[arrayShape - 1])]
        )

        # for i in range(arrayShape):
        #     nodeFeatures.append(
        #         list(array.coord[i]) +
        #         [atomsDict[array.atom_name[i]]] +
        #         [elementsDict[array.element[i]]] +
        #         [array.res_id[i]] +
        #         [residualesDict[array.res_name[i]]] +
        #         [float(array.hetero[i])]
        #     )

        nodeFeaturesT = torch.tensor(nodeFeatures, dtype=torch.float)
        edge_indexT = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=nodeFeaturesT, edge_index=edge_indexT)

        return data


set_seed(23)

fileDir = '/mnt/ssd1/prog/pdbFiles'
dataList = os.listdir(fileDir)
dataList = dataList[:20]
validatePart = 0.3
batchSize = 10
epochsNum = 5
numWorkers = 12

validateLength = int(len(dataList) * validatePart)
dataSizes = [len(dataList) - validateLength, validateLength]
dataTrainRaw, dataValidateRaw = torch.utils.data.random_split(dataList, dataSizes)

dataTrain = readData(fileDir, dataTrainRaw)
dataValidate = readData(fileDir, dataValidateRaw)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
# trainLoader = torch.utils.data.DataLoader(dataTrain, batch_size=batchSize,
#                                           num_workers=numWorkers, shuffle=True)
# validateLoader = torch.utils.data.DataLoader(dataValidate, batch_size=batchSize,
#                                              num_workers=numWorkers, shuffle=True)

trainLoader = torch_geometric.data.DataLoader(dataTrain, batch_size=batchSize,
                                              num_workers=numWorkers, shuffle=True)
validateLoader = torch_geometric.data.DataLoader(dataValidate, batch_size=batchSize,
                                                 num_workers=numWorkers, shuffle=True)

model = AutoEncoder()
# model = model.double()
lossType = nn.MSELoss()
# lossType = f.nll_loss
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

model = model.to(device)

for epoch in range(epochsNum):
    print('Epoch {}/{}:'.format(epoch, epochsNum - 1), flush=True)

    model.train()
    for data in trainLoader:
        # print(data.shape)

        # data = torch.stack((data,)).to(device)
        data = data.to(device)
        preds = model(data)

        # print(preds.shape)

        loss = lossType(preds, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.4f}'.format(float(loss)), flush=True)

    model.eval()
    for data in validateLoader:
        # data = torch.stack((data,)).to(device)
        data = data.to(device)
        preds = model(data)
        loss = lossType(preds, data)
    print('Validation Loss: {:.4f}'.format(float(loss)), flush=True)
