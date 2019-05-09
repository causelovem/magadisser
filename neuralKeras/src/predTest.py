# import setuptools
import os
# import sys
import numpy as np
import math

# import keras
from keras.models import Sequential, load_model
# from keras.utils import plot_model

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./pred/matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

quan = 16
weig = [4.04, 0.78, 0.46, 0.42, 0.41, 0.32]
print('> Readind matrix data...')
for file in matrixFiles:
    fileIn = open("./pred/matrix/" + file, "r")

    matrix = fileIn.readlines()
    dim = len(matrix)

    for i in range(dim):
        matrix[i] = matrix[i][:-2].split(' ')
        # for j in range(len(matrix[i])):
        #     matrix[i][j] = int(matrix[i][j])

        pairList = []
        for j in range(dim):
            pairList.append((j, int(matrix[i][j])))

        pairList.sort(key=lambda x: x[1])

        for j in range(dim):
            if (pairList[j][1] == 0):
                matrix[i][pairList[j][0]] = 0
            else:
                matrix[i][pairList[j][0]] = j + 1

    block = dim // quan
    sque = []
    for k in range(quan):
        newStr = []
        sum = 0
        for t in range(quan):
            for i in range(block):
                for j in range(block):
                    sum += matrix[i + t * block][j + k * block]
            sum /= block * block
            newStr.append(sum)
        sque.append(newStr)

    for i in range(quan):
        size = quan
        while size > 1:
            size //= 2
            tmp = []
            for j in range(size):
                sque[i][j] = (sque[i][2 * j] + sque[i][2 * j + 1]) / 2
                tmp.append(sque[i][j] - sque[i][2 * j + 1])
            for j in range(len(tmp)):
                sque[i][j + size] = tmp[j]

    for i in range(quan):
        size = quan
        while size > 1:
            size //= 2
            tmp = []
            for j in range(size):
                sque[j][i] = (sque[2 * j][i] + sque[2 * j + 1][i]) / 2
                tmp.append(sque[j][i] - sque[2 * j + 1][i])
            for j in range(len(tmp)):
                sque[j + size][i] = tmp[j]

    for i in range(quan):
        for j in range(quan):
            sque[i][j] *= weig[int(min(max(math.log(i + 1, 2), math.log(j + 1, 2)), 5))]

    # for i in range(dim):
    #     matrix[i] = matrix[i][:-2].split(' ')
    #     for j in range(len(matrix[i])):
    #         matrix[i][j] = int(matrix[i][j])

    # for i in range(dim):
    #     pairList = []
    #     for j in range(dim):
    #         pairList.append((j, int(matrix[j][i])))

    #     pairList.sort(key=lambda x: x[1])

    #     for j in range(dim):
    #         if (pairList[j][1] == 0):
    #             matrix[j][pairList[i][0]] = 0
    #         else:
    #             matrix[j][pairList[i][0]] = j + 1

    # tmp = np.array(matrix)
    tmp = np.array(sque)
    # print(sque)
    # print('\r\n')
    tmp = np.expand_dims(tmp, axis=2)
    matrixList.append(tmp)

    fileIn.close()

matrixVec = np.array(matrixList)
# matrixDim = int(matrixVec.shape[1])
matrixDim = 256
numOfSet = int(matrixVec.shape[0])

print(matrixVec.shape)

max = 0
if (matrixDim <= 8):
    max = 2
elif ((matrixDim <= 16) or (matrixDim <= 32) or (matrixDim <= 64)):
    max = 4
elif ((matrixDim <= 128) or (matrixDim <= 256) or (matrixDim <= 512)):
    max = 8
elif ((matrixDim <= 1024) or (matrixDim <= 2048)):
    max = 16

mappingList = []
mappingFiles = os.listdir("./pred/test")
mappingFiles.sort(key=lambda x: int(x[7:]))

step = 1.0 / (max - 1.0)

print('> Readind mapping data...')
for file in mappingFiles:
    fileIn = open("./pred/test/" + file, "r")

    mapping = fileIn.readlines()
    dim = len(mapping)

    # for i in range(dim):
    #     mapping[i] = mapping[i][:-1].split(' ')
    #     strDim = len(mapping[i])
    #     for j in range(strDim):
    #         if (mapping[i][j] != '0'):
    #             mapping[i][j] = 1.0 / int(mapping[i][j])
    #         else:
    #             mapping[i][j] = int(mapping[i][j])
    # tmp = np.array(mapping)
    # mappingList.append(tmp)

    # fileIn.close()

    for i in range(dim):
        mapping[i] = mapping[i][:-1].split(' ')
        strDim = len(mapping[i])
        for j in range(strDim):
            # mapping[i][j] = int(mapping[i][j]) * step
            # mapping[i][j] = int(mapping[i][j]) / 10
            mapping[i][j] = int(mapping[i][j]) / max
            # mapping[i][j] = int(mapping[i][j])
    tmp = np.array(mapping)
    mappingList.append(tmp)

    fileIn.close()

mappingVec = np.array(mappingList)
mappingVec = mappingVec.reshape(numOfSet, matrixDim * 4)

# sys.exit(0)

print('> Preparing for prediction...')
lenMapStr = 4
model = Sequential()
model = load_model('./nets/net1.h5')
# model = load_model('./nets/net2.h5')
# model = load_model('./nets/netnet.h5')
# plot_model(model, to_file='model.png')

# max = 0
# if (matrixDim <= 8):
#     max = 2
# elif ((matrixDim <= 16) or (matrixDim <= 32) or (matrixDim <= 64)):
#     max = 4
# elif ((matrixDim <= 128) or (matrixDim <= 256) or (matrixDim <= 512)):
#     max = 8
# elif ((matrixDim <= 1024) or (matrixDim <= 2048)):
#     max = 16


print('> Predict on test data...')
for i in range(len(matrixVec)):
    pred = model.predict(matrixVec[i:i + 1])
    # print("./pred/prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

    # for j in range(matrixDim):
    #     for k in range(lenMapStr):
    #         if (abs(pred[0][j * lenMapStr + k] - 0) > 0.00001):
    #             tmp = 1 / pred[0][j * lenMapStr + k]
    #             if (round(tmp) > max - 1):
    #                 fileOut.write(str(int(0)) + ' ')
    #             else:
    #                 fileOut.write(str(int(round(tmp))) + ' ')
    #         else:
    #             fileOut.write(str(int(0)) + ' ')
    #     fileOut.write('\n')

    # fileOut.close()

    for j in range(matrixDim):
        for k in range(lenMapStr):
            if (abs(pred[0][j * lenMapStr + k] - 0) > 0.001):
                # tmp = int(round(pred[0][j * lenMapStr + k] / step))
                # tmp = int(round(pred[0][j * lenMapStr + k] * 10))
                tmp = int(round(pred[0][j * lenMapStr + k] * max))
                # tmp = int(round(pred[0][j * lenMapStr + k]))
                if (tmp > max - 1):
                    fileOut.write(str(int(0)) + ' ')
                else:
                    fileOut.write(str(int(tmp)) + ' ')
            else:
                fileOut.write(str(int(0)) + ' ')
        fileOut.write('\n')

    fileOut.close()

score = model.evaluate(matrixVec, mappingVec, batch_size=50)
print(score)
