# по обученой модели переводим графы или pdb-файлы в вектора

import numpy as np
import torch
import os
import sys
from gaeModel import AutoEncoder
import config as cfg
import pdb2Gdata_v5 as p2d
from tqdm import tqdm
# from sklearn.cluster import KMeans, MeanShift
# from matplotlib import pyplot as plt


if len(sys.argv) != 2:
    print('ERROR: Check your command string')
    print('Usage: python3 pred.py <absoluteFileName>')
    sys.exit(-1)

structure = None
if cfg.pdbFile:
    structure = p2d.pdb2Gdata(sys.argv[1])
else:
    structure = torch.load(sys.argv[1])

# считываем модель
device = cfg.device
model = AutoEncoder()
model.load_state_dict(torch.load(os.path.join(cfg.modelsDir, 'testModel.pt')))
model = model.to(device)
model.eval()

# применяем энкодер
structure = structure.to(device)
pred = model.encoder(structure.x, structure.edge_index).to('cpu').detach().numpy()
pred = pred.sum(axis=0) / len(pred)

# считываем существующие вектора
vectorDir = cfg.vectorDir
dataList = os.listdir(vectorDir)
dataListNp = np.array(dataList)
len(dataList)

vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])

# ret = os.getcwd()
# os.chdir(vectorDir)
# vectors = np.array([np.load(file) for file in tqdm(dataList)])
# os.chdir(ret)

# считаем расстояния
# dist = np.array([np.linalg.norm(pred - vec) for vec in tqdm(vectors)])
dist = np.linalg.norm(vectors - pred, axis=1)
distNorm = dist / dist.max()

distSortIndeces = np.argsort(dist)

resSorted = dataListNp[distSortIndeces]
# print(res[:20])

# выводим те, которые нам нужны (коэффициенты можно варировать)
mask = distNorm <= 0.01
res = dataListNp[mask]
res.shape

# можно построить карты корреляции
# import seaborn as sns
# corr = np.corrcoef(rawAttentVectors, rowvar=False)
# ax = sns.heatmap(corr)
# plt.show()
