# import setuptools
import os
import sys
import numpy as np
import math

# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D
# from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
# from keras.callbacks import EarlyStopping
# from keras.optimizers import SGD
# from keras.utils import plot_model, normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
# from keras.regularizers import l2

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

persent = -1
quan = 16
weig = [4.04, 0.78, 0.46, 0.42, 0.41, 0.32]
print('> Readind matrix data...')
for file in matrixFiles:
    persent += 1
    print(str(round(persent * 100 / len(matrixFiles), 1)) + '%', end='')
    print('\r', end='')

    fileIn = open("./matrix/" + file, "r")

    matrix = fileIn.readlines()
    dim = len(matrix)

    for i in range(dim):
        matrix[i] = matrix[i][:-2].split(' ')
        for j in range(len(matrix[i])):
            matrix[i][j] = int(matrix[i][j])

        pairList = []
        for j in range(dim):
            pairList.append((j, int(matrix[i][j])))

        pairList.sort(key=lambda x: x[1])

        for j in range(dim):
            if (pairList[j][1] == 0):
                matrix[i][pairList[j][0]] = 0
            else:
                matrix[i][pairList[j][0]] = j + 1

    aa = np.array(matrix)
    plt.imshow(aa)
    plt.colorbar()
    plt.show()

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

    aa = np.array(sque)
    plt.imshow(aa)
    plt.colorbar()
    plt.show()
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

    aa = np.array(sque)
    plt.imshow(aa)
    plt.colorbar()
    plt.show()
    # for i in range(quan):
    #     for j in range(quan):
    #         sque[i][j] *= weig[int(min(max(math.log(i + 1, 2), math.log(j + 1, 2)), 5))]

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

# matrixVec = normalize(matrixVec, 2)

# print(matrixVec.shape)
