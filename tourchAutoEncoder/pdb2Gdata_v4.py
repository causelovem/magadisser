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
    chainIdUnique = np.uniqueu(array.chain_id)

    data = {}
    sseMaskDict = dict([(chain, {}) for chain in chainIdUnique])
    for chain in chainIdUnique:
        # берем текущую цепь
        oneChainArray = array[array.chain_id == chain]

        # "маска" остатков СА атомов
        resMask = oneChainArray[oneChainArray.atom_name == 'CA'].res_id

        # НЕ считаем вторичную стуктуру, если в цепи нет (или мало) CA атомов
        if resMask.shape[0] < 5:
            continue

        # вторичная структура используя алгоритм DSSP для каждой цепи
        sse = dssp.DsspApp.annotate_sse(oneChainArray)

        # если sse короче маски, то расширим
        tmp = resMask.shape[0] - sse.shape[0]
        if tmp > 0:
            sse = np.append(sse, ['Null'] * tmp)

        # для каждой цепи, для каждого остатка - вторичная структура
        for resId, sseId in zip(resMask, sse):
            sseMaskDict[chain][resId] = sseId

        # матрица смежности
        cellList = struc.CellList(oneChainArray, cell_size=cfg.threshold)
        adjMatrix = cellList.create_adjacency_matrix(cfg.threshold)

        # вычитаем цетроиду - смещаем цетр белка в точку (0, 0, 0) (для нормировки признака)
        oneChainArray.coord -= oneChainArray.coord.mean(axis=0)

        # длина максимального вектора (для нормировки признака)
        maxNorm = max([np.linalg.norm(point) for point in oneChainArray.coord])
        oneChainArray.coord /= maxNorm

        # максимальный температурный фактор (для нормировки признака)
        maxBFactor = oneChainArray.b_factor.max()
        oneChainArray.b_factor /= maxBFactor

        edgeIndex = [[], []]
        nodeFeatures = []

        # переводим матрицу смежности в COO и собираем признаки
        arrayShp = oneChainArray.shape[0]
        for i in range(arrayShp - 1):
            for j in range(i + 1, arrayShp):
                if adjMatrix[i][j]:
                    edgeIndex[0].append(i)
                    edgeIndex[1].append(j)

            nodeFeatures.append(
                list(oneChainArray.coord[i]) +
                [oneChainArray.b_factor[i],
                 float(oneChainArray.hetero[i]),
                 oneChainArray.occupancy[i]] +
                atomsDict.get(oneChainArray.atom_name[i], atomsDict['Null']) +
                residualesDict.get(oneChainArray.res_name[i], residualesDict['Null']) +
                ssesTypeDict.get(sseMaskDict[oneChainArray.chain_id[i]].get(oneChainArray.res_id[i],
                                                                            'Null'),
                                 ssesTypeDict['Null'])
            )
        nodeFeatures.append(
            list(oneChainArray.coord[-1]) +
            [oneChainArray.b_factor[-1],
             float(oneChainArray.hetero[-1]),
             oneChainArray.occupancy[-1]] +
            atomsDict.get(oneChainArray.atom_name[-1], atomsDict['Null']) +
            residualesDict.get(oneChainArray.res_name[-1], residualesDict['Null']) +
            ssesTypeDict.get(sseMaskDict[oneChainArray.chain_id[-1]].get(oneChainArray.res_id[-1],
                                                                         'Null'),
                             ssesTypeDict['Null'])
        )

        # графовый формат
        data[chain] = Data(x=torch.tensor(nodeFeatures, dtype=torch.float),
                           edge_index=torch.tensor(edgeIndex, dtype=torch.long))

    # сохраняем все графы в отдельные файлы
    if saveDir:
        for chain, graph in data.items():
            fileNameSplit = fileName.split('.')
            fileNameSplit[0] += chain
            torch.save(graph, os.path.join(saveDir, '.'.join(fileNameSplit)))

    # возвращаем словарь
    return data


if __name__ == '__main__':
    import sys

    argvLen = len(sys.argv)
    if argvLen not in (3, 4):
        print('ERROR: Check your command string')
        print('Usage: python3 pdb2Gdata.py <dirName> <fileName> [<saveDir>]')
        sys.exit(-1)

    pdb2Gdata(sys.argv[1], sys.argv[2], sys.argv[3] if argvLen == 4 else False)
