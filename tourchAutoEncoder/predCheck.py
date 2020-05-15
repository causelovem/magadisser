import numpy as np
import os
import config as cfg
from tqdm import tqdm


vectorDir = cfg.vectorDir
rawVectorDir = cfg.rawVectorDir
attentVectorDir = cfg.attentVectorDir
rawAttentVectorDir = cfg.rawAttentVectorDir

dataList = os.listdir(vectorDir)
dataListNp = np.array([file[3:-8] for file in tqdm(dataList)])

# vectors = np.array([np.load(os.path.join(vectorDir, file)) for file in tqdm(dataList)])
vectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/vectors.npy')

# rawVectors = np.array([np.load(os.path.join(rawVectorDir, file)) for file in tqdm(dataList)])
rawVectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/rawVectors.npy')

# rawAttentVectors = np.array([np.load(os.path.join(rawAttentVectorDir, file)) for file in tqdm(dataList)])
rawAttentVectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/rawAttentVector.npy')

# attentVectors = np.array([np.load(os.path.join(attentVectorDir, file)) for file in tqdm(dataList)])
attentVectors = np.load('F:/prog/magadisser/tourchAutoEncoder/data/attentVectors.npy')

pred = vectors[49848]
attentPred = attentVectors[49848]
rawPred = rawVectors[49848]
rawAttentPred = rawAttentVectors[49848]
# 1on7A checkAcc(30000) - ПЛОХО, 21% / 46%
# 1fagD checkAcc(10000) - НОРМАЛЬНО, 67% / 75%
# 1agdB checkAcc(1000) - ОЧЕНЬ ХОРОШО, 95% / 94%
# 6esfD checkAcc(300000) - ПЛОХО, 11% / 28%
# 1xycB checkAcc(50000) - НОРМАЛЬНО, 66% / 74%
# 101mA checkAcc(0) - ОЧЕНЬ ХОРОШО, 100% / 100%
# 3hsiA checkAcc(123478) - ОЧЕНЬ ПЛОХО  / 2%
# 1f8tL checkAcc(9879) - НОРМАЛЬНО,  / 65%
# 104mA checkAcc(7) - ОЧЕНЬ ХОРОШО,  / 100%
# 1a8kD checkAcc(652) -  / 90%
# 1xwrC checkAcc(49848) - ОЧЕНЬ ПЛОХО  / 2%
# 1a75B checkAcc(569) - ПЛОХО  / 23%
# 3gumB checkAcc(120919) - НОРМАЛЬНО / 50%
# 2ax1B checkAcc(56678) - ХОРОШО / 71%

dist = np.linalg.norm(vectors - pred, axis=1)
attentDist = np.linalg.norm(attentVectors - attentPred, axis=1)
rawDist = np.linalg.norm(rawVectors - rawPred, axis=1)
rawAttentDist = np.linalg.norm(rawAttentVectors - rawAttentPred, axis=1)

resSorted = dataListNp[np.argsort(dist)]
attentResSorted = dataListNp[np.argsort(attentDist)]
rawResSorted = dataListNp[np.argsort(rawDist)]
rawAttentResSorted = dataListNp[np.argsort(rawAttentDist)]

resSorted[:50]
attentResSorted[:50]
rawResSorted[:50]
rawAttentResSorted[:50]


def findPos(pdbIdC):
    pos = {}
    pos['resSorted'] = np.argwhere(resSorted == pdbIdC)[0][0]
    pos['attentResSorted'] = np.argwhere(attentResSorted == pdbIdC)[0][0]
    pos['rawResSorted'] = np.argwhere(rawResSorted == pdbIdC)[0][0]
    pos['rawAttentResSorted'] = np.argwhere(rawAttentResSorted == pdbIdC)[0][0]
    print(pos)


def checkAcc(pos):
    # norms
    dist = np.linalg.norm(vectors - vectors[pos], axis=1)
    attentDist = np.linalg.norm(attentVectors - attentVectors[pos], axis=1)
    rawDist = np.linalg.norm(rawVectors - rawVectors[pos], axis=1)
    rawAttentDist = np.linalg.norm(rawAttentVectors - rawAttentVectors[pos], axis=1)
    # sort
    resSorted = dataListNp[np.argsort(dist)]
    attentResSorted = dataListNp[np.argsort(attentDist)]
    rawResSorted = dataListNp[np.argsort(rawDist)]
    rawAttentResSorted = dataListNp[np.argsort(rawAttentDist)]
    # compare
    # testSet = set(resSorted[dist[dist <= dist.mean()].shape[0]:])
    f = open('data/siteSets/{}.txt'.format(dataListNp[pos]), 'r')
    siteSet = set([s[:-1] for s in f])
    f.close()
    accDict = {}
    accDict['res'] = len(set(resSorted[:100]) & siteSet)
    accDict['resAttent'] = len(set(attentResSorted[:100]) & siteSet)
    accDict['raw'] = len(set(rawResSorted[:100]) & siteSet)
    accDict['rawAttent'] = len(set(rawAttentResSorted[:100]) & siteSet)
    print(accDict)
