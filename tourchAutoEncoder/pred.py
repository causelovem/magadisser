import numpy as np
import torch
import os
import sys
from gaeModel import AutoEncoder
import config as cfg
import pdb2Gdata as p2d
from tqdm import tqdm
# from sklearn.cluster import KMeans, MeanShift
# from matplotlib import pyplot as plt


if len(sys.argv) != 2:
    print('ERROR: Check your command string')
    print('Usage: python3 pred.py <fileName>')
    sys.exit(-1)

structure = None
if cfg.pdbFile:
    structure = p2d.pdb2Gdata('F:/prog/magadisser/tourchAutoEncoder/data/pdbFiles', sys.argv[1])
else:
    structure = torch.load(os.path.join('F:/prog/magadisser/tourchAutoEncoder/data/pdbFiles', sys.argv[1]))

device = cfg.device
model = AutoEncoder()
model.load_state_dict(torch.load(os.path.join(cfg.modelsDir, 'testModel.pt')))
model = model.to(device)
model.eval()

structure = structure.to(device)
pred = model.encoder(structure.x, structure.edge_index).to('cpu')
pred = pred.detach().sum(dim=0) / len(pred)
pred = pred.numpy()

vectorDir = cfg.vectorDir
dataList = os.listdir(vectorDir)
len(dataList)

vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])

# ret = os.getcwd()
# os.chdir(vectorDir)
# vectors = np.array([np.load(file) for file in tqdm(dataList)])
# os.chdir(ret)

dist = np.array([np.linalg.norm(pred - vec) for vec in tqdm(vectors)])

# corr = np.corrcoef(vectors, rowvar=False)
# ax = sns.heatmap(corr)
# plt.show()
