import numpy as np
import os
import config as cfg
from tqdm import tqdm


vectorDir = cfg.vectorDir
rawVectorDir = cfg.rawVectorDir
attentVectorDir = cfg.attentVectorDir

dataList = os.listdir(vectorDir)
dataListNp = np.array(dataList)

rawDataList = os.listdir(rawVectorDir)
rawDataListNp = np.array(rawDataList)

attentDataList = os.listdir(attentVectorDir)
attentDataListNp = np.array(attentDataList)

vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])

rawVectors = np.array([np.load(os.path.join(rawVectorDir, file)) for file in tqdm(rawDataList)])

attentVectors = np.array([np.load(os.path.join(attentVectorDir, file)) for file in tqdm(attentDataList)])

pred = vectors[200000]
rawPred = rawVectors[200000]
attentPred = attentVectors[200000]
# 4netA

dist = np.linalg.norm(vectors - pred, axis=1)
rawDist = np.linalg.norm(rawVectors - rawPred, axis=1)
attentDist = np.linalg.norm(attentVectors - attentPred, axis=1)


distSortIndeces = np.argsort(dist)
dataListNp[distSortIndeces][:50]
rawDistSortIndeces = np.argsort(rawDist)
rawDataListNp[rawDistSortIndeces][:50]
attentDistSortIndeces = np.argsort(attentDist)
attentDataListNp[attentDistSortIndeces][:50]

resSorted = dataListNp[distSortIndeces]
rawResSorted = rawDataListNp[rawDistSortIndeces]
attentResSorted = attentDataListNp[attentDistSortIndeces]


def findPos(pdbIdC):
    pos = {}
    pos['resSorted'] = np.argwhere(resSorted == 'pdb{}.pdb.npy'.format(pdbIdC))[0][0]
    pos['rawResSorted'] = np.argwhere(rawResSorted == 'pdb{}.pdb.npy'.format(pdbIdC))[0][0]
    pos['attentResSorted'] = np.argwhere(attentResSorted == 'pdb{}.pdb.npy'.format(pdbIdC))[0][0]
    print(pos)
