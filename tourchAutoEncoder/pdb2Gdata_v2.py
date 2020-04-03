import os
import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.application.dssp as dssp
from torch_geometric.data import Data
import config as cfg


def list2OHEdict(inList):
    inLen = len(inList)
    zeros = [0] * inLen

    resDict = {}
    for i in range(inLen):
        ohe = zeros.copy()
        ohe[i] = 1.0
        resDict[inList[i]] = ohe

    return resDict


residuales = [
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'CYS',
    'GLN',
    'GLU',
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL',
    'Null'
]

residualesDict = list2OHEdict(residuales)

atoms = [
    'C',
    'C2',
    'CA',
    'CB',
    'CD',
    'CD1',
    'CD2',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'CG',
    'CG1',
    'CG2',
    'CH2',
    'CZ',
    'CZ2',
    'CZ3',
    'H',
    'HA',
    'HB',
    'HB2',
    'HB3',
    'HD2',
    'HG',
    'HG2',
    'HG21',
    'HG22',
    'HG23',
    'HG3',
    'N',
    'ND1',
    'ND2',
    'NE',
    'NE1',
    'NE2',
    'NH1',
    'NH2',
    'NZ',
    'O',
    'OD1',
    'OD2',
    'OE1',
    'OE2',
    'OG',
    'OG1',
    'OH',
    'P',
    'SD',
    'SG',
    'Null'
]

atomsDict = list2OHEdict(atoms)

ssesTypeDssp = [
    'C',
    'H',
    'B',
    'E',
    'G',
    'I',
    'T',
    'S',
    'Null'
]

ssesTypeDict = list2OHEdict(ssesTypeDssp)


def pdb2Gdata(dirName, fileName, saveDir=False):
    # print(os.path.join(dirName, fileName))
    array = strucio.load_structure(os.path.join(dirName, fileName),
                                   # extra_fields=['atom_id', 'b_factor', 'occupancy', 'charge'],
                                   extra_fields=['b_factor', 'occupancy'],
                                   model=1)

    # уникальные цепи
    chainIdUnique = []
    for chain in array.chain_id:
        if chain not in chainIdUnique:
            chainIdUnique.append(chain)

    # вторичная структура используя алгоритм DSSP
    sse = dssp.DsspApp.annotate_sse(array)

    # "маски" цепи и остатки СА атомов
    chainMask = array[array.atom_name == 'CA'].chain_id
    resMask = array[array.atom_name == 'CA'].res_id

    # если sse короче масок, то расширим
    tmp = resMask.shape[0] - sse.shape[0]
    if tmp > 0:
        sse = np.append(sse, ['Null'] * tmp)

    # для каждой цепи, для каждого остатка - вторичная структура
    sseMaskDict = dict([(chain, {}) for chain in chainIdUnique])
    for chainId, resId, sseId in zip(chainMask, resMask, sse):
        sseMaskDict[chainId][resId] = sseId

    # матрица смежности
    cell_list = struc.CellList(array, cell_size=cfg.threshold)
    adj_matrix = cell_list.create_adjacency_matrix(cfg.threshold)

    # (adj_matrix[adj_matrix == True].shape[0] - 5385) / 2
    edge_index = [[], []]
    nodeFeatures = []

    # переводим матрицу смежности в COO и собираем признаки
    arrayShp = array.shape[0]
    for i in range(arrayShp - 1):
        for j in range(i + 1, arrayShp):
            if adj_matrix[i][j]:
                edge_index[0].append(i)
                edge_index[1].append(j)

        nodeFeatures.append(
            list(array.coord[i]) +
            [array.res_id[i],
             array.b_factor[i],
             float(array.hetero[i]),
             array.occupancy[i]] +
            atomsDict.get(array.atom_name[i], atomsDict['Null']) +
            residualesDict.get(array.res_name[i], residualesDict['Null']) +
            ssesTypeDict.get(sseMaskDict[array.chain_id[i]].get(array.res_id[i],
                                                                'Null'),
                             ssesTypeDict['Null'])
        )
    nodeFeatures.append(
        list(array.coord[arrayShp - 1]) +
        [array.res_id[arrayShp - 1],
         array.b_factor[arrayShp - 1],
         float(array.hetero[arrayShp - 1]),
         array.occupancy[arrayShp - 1]] +
        atomsDict.get(array.atom_name[arrayShp - 1], atomsDict['Null']) +
        residualesDict.get(array.res_name[arrayShp - 1], residualesDict['Null']) +
        ssesTypeDict.get(sseMaskDict[array.chain_id[arrayShp - 1]].get(array.res_id[arrayShp - 1],
                                                                       'Null'),
                         ssesTypeDict['Null'])
    )

    # графовый формат
    # nodeFeaturesT = torch.tensor(nodeFeatures, dtype=torch.float)
    # edge_indexT = torch.tensor(edge_index, dtype=torch.long)
    # data = Data(x=nodeFeaturesT, edge_index=edge_indexT)
    data = Data(x=torch.tensor(nodeFeatures, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))

    if saveDir:
        torch.save(data, os.path.join(saveDir, fileName))

    return data


if __name__ == '__main__':
    import sys

    argvLen = len(sys.argv)
    if argvLen not in (3, 4):
        print('ERROR: Check your command string')
        print('Usage: python3 pdb2Gdata.py <dirName> <fileName> [<saveDir>]')
        sys.exit(-1)

    pdb2Gdata(sys.argv[1], sys.argv[2], sys.argv[3] if argvLen == 4 else False)
