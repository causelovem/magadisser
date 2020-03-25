import os
import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io as strucio
from torch_geometric.data import Data
import config as cfg


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


def pdb2Gdata(dirName, fileName, saveDir=False):
    # print(os.path.join(dirName, fileName))
    array = strucio.load_structure(os.path.join(dirName, fileName),
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

    cell_list = struc.CellList(array, cell_size=cfg.threshold)
    adj_matrix = cell_list.create_adjacency_matrix(cfg.threshold)

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
    edge_indexT = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=nodeFeaturesT, edge_index=edge_indexT)

    if saveDir:
        torch.save(data, os.path.join(saveDir, fileName))

    # print(os.path.join(dirName, fileName), type(data), type(data.x), type(data.edge_index))
    return data


if __name__ == '__main__':
    import sys

    argvLen = len(sys.argv)
    if argvLen not in (3, 4):
        print('ERROR: Check your command string')
        print('Usage: python3 pdb2Gdata.py <dirName> <fileName> [<saveDir>]')
        sys.exit(-1)

    pdb2Gdata(sys.argv[1], sys.argv[2], sys.argv[3] if argvLen == 4 else False)
