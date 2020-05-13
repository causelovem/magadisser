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
    'HOH',
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

ssesTypeDssp = [
    'C',
    'H',
    'B',
    'E',
    'G',
    'I',
    'T',
    'S'
]

ssesTypeDict = list2OHEdict(ssesTypeDssp)

posDim = 4
posMax = 10000
posEncoding = np.array([
    [pos / np.power(10000, 2 * i / posDim) for i in range(posDim)]
    if pos != 0 else np.zeros(posDim) for pos in range(posMax)])

posEncoding[1:, 0::2] = np.sin(posEncoding[1:, 0::2])
posEncoding[1:, 1::2] = np.cos(posEncoding[1:, 1::2])

posEncoding = np.float32(posEncoding)


def pdb2Gdata(dirName, fileName, saveDir=False):
    array = strucio.load_structure(os.path.join(dirName, fileName), model=1)

    # уникальные цепи
    chainIdUnique = np.unique(array.chain_id)

    data = {}
    # для каждой цепи
    for chain in chainIdUnique:
        sseMaskDict = {}

        # берем текущую цепь, исключаем heatem атомы (== numpy.False)
        oneChainArray = array[(array.chain_id == chain) & (array.hetero == False)]

        # только СА атомы
        backbone = oneChainArray[oneChainArray.atom_name == 'CA']

        backboneShp = backbone.shape[0]
        # НЕ считаем вторичную стуктуру, если в цепи нет (или мало) CA атомов
        if backboneShp < 5:
            continue

        # вторичная структура используя алгоритм DSSP
        sse = dssp.DsspApp.annotate_sse(oneChainArray)

        # если sse короче маски, то расширим
        tmp = backboneShp - sse.shape[0]
        if tmp > 0:
            sse = np.append(sse, ['C'] * tmp)

        # для каждого остатка - вторичная структура
        for resId, sseId in zip(backbone.res_id, sse):
            sseMaskDict[resId] = sseId

        # матрица смежности
        cellList = struc.CellList(backbone, cell_size=cfg.threshold)
        adjMatrix = cellList.create_adjacency_matrix(cfg.threshold)

        # вычитаем центроиду - смещаем центр белка в точку (0, 0, 0) (для нормировки признака)
        backbone.coord -= backbone.coord.mean(axis=0)

        # длина максимального вектора (для нормировки признака)
        maxNorm = np.linalg.norm(backbone.coord, axis=1).max()
        if maxNorm != 0:
            backbone.coord /= maxNorm

        edgeIndex = [[], []]
        nodeFeatures = []

        # переводим матрицу смежности в COO и собираем признаки
        for i in range(backboneShp - 1):
            for j in range(i + 1, backboneShp):
                if adjMatrix[i][j]:
                    edgeIndex[0].append(i)
                    edgeIndex[1].append(j)

            nodeFeatures.append([
                *list(backbone.coord[i]),
                *residualesDict.get(backbone.res_name[i], residualesDict['Null']),
                *ssesTypeDict.get(sseMaskDict.get(backbone.res_id[i], 'C')),
                *list(posEncoding[i])
            ])
        nodeFeatures.append([
            *list(backbone.coord[-1]),
            *residualesDict.get(backbone.res_name[-1], residualesDict['Null']),
            *ssesTypeDict.get(sseMaskDict.get(backbone.res_id[-1], 'C')),
            *list(posEncoding[backboneShp - 1])
        ])

        # графовый формат
        data[chain] = Data(x=torch.tensor(nodeFeatures, dtype=torch.float),
                           edge_index=torch.tensor(edgeIndex, dtype=torch.long))

    # сохраняем все графы в отдельные файлы
    if saveDir:
        for chain, graph in data.items():
            fileNameSplit = fileName.split('.')
            # приписываем к названию файла название цепи
            fileNameSplit[0] += chain
            torch.save(graph, os.path.join(saveDir, '.'.join(fileNameSplit)))

    # возвращаем словарь
    return data


if __name__ == '__main__':
    import sys

    argvLen = len(sys.argv)
    if argvLen not in (3, 4):
        print('ERROR: Check your command string')
        print('Usage: python3 pdb2Gdata_v5.py <dirName> <fileName> [<saveDir>]')
        sys.exit(-1)

    pdb2Gdata(sys.argv[1], sys.argv[2], sys.argv[3] if argvLen == 4 else False)
