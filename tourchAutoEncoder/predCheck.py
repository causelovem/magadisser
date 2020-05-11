import numpy as np
import os
import config as cfg
from tqdm import tqdm


vectorDir = cfg.vectorDir
rawVectorDir = cfg.rawVectorDir
attentVectorDir = cfg.attentVectorDir

dataList = os.listdir(vectorDir)
dataListNp = np.array([file[3:-8] for file in tqdm(dataList)])
# dataListNp = np.array(dataList)


# rawDataList = os.listdir(rawVectorDir)
# rawDataListNp = np.array(rawDataList)

# attentDataList = os.listdir(attentVectorDir)
# attentDataListNp = np.array(attentDataList)

# vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])
vectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/vectors.npy')

# rawVectors = np.array([np.load(os.path.join(rawVectorDir, file)) for file in tqdm(dataList)])
rawVectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/rawVectors.npy')

# attentVectors = np.array([np.load(os.path.join(attentVectorDir, file)) for file in tqdm(dataList)])
attentVectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/attentVectors.npy')

pred = vectors[30000]
rawPred = rawVectors[30000]
attentPred = attentVectors[30000]
# 1on7A [30000] - ПЛОХО, 21%
# 1fagD [10000] - НОРМАЛЬНО, 67%
# 1agdB [1000] - ОЧЕНЬ ХОРОШО, 95%
# 6esfD [300000] - ПЛОХО, 11%
# 1xycB [50000] - НОРМАЛЬНО, 66%
# 101mA [0] - ОЧЕНЬ ХОРОШО, 100%

dist = np.linalg.norm(vectors - pred, axis=1)
rawDist = np.linalg.norm(rawVectors - rawPred, axis=1)
attentDist = np.linalg.norm(attentVectors - attentPred, axis=1)


distSortIndeces = np.argsort(dist)
dataListNp[distSortIndeces][:50]
rawDistSortIndeces = np.argsort(rawDist)
dataListNp[rawDistSortIndeces][:50]
attentDistSortIndeces = np.argsort(attentDist)
dataListNp[attentDistSortIndeces][:50]

resSorted = dataListNp[distSortIndeces]
rawResSorted = dataListNp[rawDistSortIndeces]
attentResSorted = dataListNp[attentDistSortIndeces]

testSet = set(resSorted[dist[dist <= dist.mean()].shape[0]:])
f = open('data/siteSet.txt', 'r')
siteSet = set([s[:-1] for s in f])
f.close()
# t = testSet & siteSet
len(testSet & siteSet)


def findPos(pdbIdC):
    pos = {}
    pos['resSorted'] = np.argwhere(resSorted == pdbIdC)[0][0]
    pos['rawResSorted'] = np.argwhere(rawResSorted == pdbIdC)[0][0]
    pos['attentResSorted'] = np.argwhere(attentResSorted == pdbIdC)[0][0]
    print(pos)
