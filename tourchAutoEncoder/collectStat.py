# "сбор" статистики

import os
import numpy as np
import biotite.structure.io as strucio
import config as cfg
from tqdm import tqdm


atomName = {}
resName = {}

dataList = os.listdir(cfg.rawFileDir)

# проходимся по всем файлам и считаем количество разных атомов и аминокислот
ret = os.getcwd()
os.chdir(cfg.rawFileDir)
for file in tqdm(dataList):
    try:
        array = strucio.load_structure(file, model=1)
        chainIdUnique = np.unique(array.chain_id)
        for chain in chainIdUnique:
            if array[(array.chain_id == chain) & (array.atom_name == 'CA')].shape[0] >= 5:
                a = array[array.chain_id == chain]
                for elem in np.array(np.unique(a.atom_name, return_counts=True)).T:
                    atomName[elem[0]] = atomName.get(elem[0], 0) + int(elem[1])
                for elem in np.array(np.unique(a.res_name, return_counts=True)).T:
                    resName[elem[0]] = resName.get(elem[0], 0) + int(elem[1])
    except Exception as e:
        print(e)
        continue
os.chdir(ret)


# переводим все в нумпай и сохраняем
atomArr = [(key, val) for key, val in atomName.items()]
resArr = [(key, val) for key, val in resName.items()]

atomArr = np.array(sorted(atomArr, key=lambda x: x[1]))
resArr = np.array(sorted(resArr, key=lambda x: x[1]))

np.save(os.path.join(cfg.statDir, 'atomStat'), atomArr)
np.save(os.path.join(cfg.statDir, 'resStat'), resArr)

# atomArr = np.load(os.path.join(cfg.statDir, 'atomStat.npy'))
# resArr = np.load(os.path.join(cfg.statDir, 'resStat.npy'))
# atomDict = dict(atomArr)
# resDict = dict(resArr)

print(atomArr[-50:])
print(resArr[-20:])
