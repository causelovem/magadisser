import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import os
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
# import biotite.application.dssp as dssp
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
    'Null'
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
    'Null'
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
    'Null'
]

atomsDict = dict([[atoms[i], float(i)] for i in range(len(atoms))])

ssesType = [
    'a',
    'b',
    'c',
    'Null'
]

ssesTypeDict = dict([[ssesType[i], float(i)] for i in range(len(ssesType))])


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

        self.conv1 = GCNConv(11, 16)
        self.conv2 = GCNConv(16, 32)

        self.unconv1 = GCNConv(32, 16)
        self.unconv2 = GCNConv(16, 11)

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


class readData(torch_geometric.data.Dataset):
    def __init__(self, fileDir, files):
        self.fileDir = fileDir
        self.files = files
        self.threshold = 7
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            # print(os.path.join(self.fileDir, self.files[index]))
            array = strucio.load_structure(os.path.join(self.fileDir, self.files[index]),
                                           # extra_fields=['atom_id', 'b_factor', 'occupancy', 'charge'],
                                           extra_fields=['b_factor', 'occupancy'],
                                           model=1)
            # if type(array) == biotite.structure.AtomArrayStack:
            #     array = array[0]

            # ca = array[array.atom_name == "CA"]
            # cell_list = struc.CellList(ca, cell_size=self.threshold)

            chain_id = []
            for chain in array.chain_id:
                if chain not in chain_id:
                    chain_id.append(chain)

            sseDict = dict([(chain, struc.annotate_sse(array, chain_id=chain)) for chain in chain_id])

            sseMaskDict = {}
            for key, value in sseDict.items():
                mask = array[(array.chain_id == key) & (array.atom_name == 'CA')].res_id
                tmp = mask.shape[0] - value.shape[0]
                if tmp > 0:
                    sseDict[key] = np.append(value, ['Null'] * tmp)

                sseMaskDict[key] = {}
                for maskId, sseId in zip(mask, sseDict[key]):
                    sseMaskDict[key][maskId] = sseId

            cell_list = struc.CellList(array, cell_size=self.threshold)
            adj_matrix = cell_list.create_adjacency_matrix(self.threshold)

            # (adj_matrix[adj_matrix == True].shape[0] - 5385) / 2
            edge_index = [[], []]

            nodeFeatures = []
            arrayShp = array.shape[0]
            for i in range(arrayShp - 1):
                for j in range(i + 1, arrayShp):
                    if adj_matrix[i][j]:
                        edge_index[0].append(i)
                        edge_index[1].append(j)

                nodeFeatures.append(
                    list(array.coord[i]) +
                    [atomsDict.get(array.atom_name[i], atomsDict['Null'])] +
                    [elementsDict.get(array.element[i], elementsDict['Null'])] +
                    [array.res_id[i]] +
                    [residualesDict.get(array.res_name[i], residualesDict['Null'])] +
                    [float(array.hetero[i])] +
                    [array.occupancy[i]] +
                    [array.b_factor[i]] +
                    [ssesTypeDict.get(sseMaskDict[array.chain_id[i]].get(array.res_id[i],
                                                                         'Null'),
                                      ssesTypeDict['Null'])]
                )
            nodeFeatures.append(
                list(array.coord[arrayShp - 1]) +
                [atomsDict.get(array.atom_name[arrayShp - 1], atomsDict['Null'])] +
                [elementsDict.get(array.element[arrayShp - 1], elementsDict['Null'])] +
                [array.res_id[arrayShp - 1]] +
                [residualesDict.get(array.res_name[arrayShp - 1], residualesDict['Null'])] +
                [float(array.hetero[arrayShp - 1])] +
                [array.occupancy[arrayShp - 1]] +
                [array.b_factor[arrayShp - 1]] +
                [ssesTypeDict.get(sseMaskDict[array.chain_id[arrayShp - 1]].get(array.res_id[arrayShp - 1],
                                                                                'Null'),
                                  ssesTypeDict['Null'])]
            )

            nodeFeaturesT = torch.tensor(nodeFeatures, dtype=torch.float)
            edge_indexT = torch.tensor(edge_index, dtype=torch.int)
            data = Data(x=nodeFeaturesT, edge_index=edge_indexT)

            # torch.save(data, 'file')
            # a = torch.load('file')

            return data
        except biotite.InvalidFileError:
            print('!!!!!!!!!!!' + os.path.join(self.fileDir, self.files[index]))


set_seed(23)

fileDir = '/mnt/ssd1/prog/pdbFiles'
dataList = os.listdir(fileDir)
dataList = dataList[:50]
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
