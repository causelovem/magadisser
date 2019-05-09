# import setuptools
import os
# import sys
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model, normalize, np_utils
# import matplotlib.pyplot as plt
# from keras.regularizers import l2

from mpi4py import MPI as mpi

np.set_printoptions(threshold=np.nan)

comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

matrixList = []
# matrixFiles = os.listdir("./matrix")
matrixFiles = os.listdir("./pred/matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

persent = -1
print('> Readind matrix data...')
for file in matrixFiles:
    persent += 1
    print(str(round(persent * 100 / len(matrixFiles), 1)) + '%', end='')
    print('\r', end='')

    # fileIn = open("./matrix/" + file, "r")
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

    tmp = np.array(matrix)
    tmp = np.expand_dims(tmp, axis=2)
    matrixList.append(tmp)

    fileIn.close()

matrixVec = np.array(matrixList)
matrixDim = int(matrixVec.shape[1])
numOfSet = int(matrixVec.shape[0])

# print(matrixVec[0])
# matrixVec = normalize(matrixVec, 2)
# print(matrixVec[0])

print(matrixVec.shape)

# sys.exit(0)

mappingList = []
# mappingFiles = os.listdir("./mapping")
mappingFiles = os.listdir("./pred/test")
mappingFiles.sort(key=lambda x: int(x[7:]))

persent = -1
print('> Readind mapping data...')
for file in mappingFiles:
    persent += 1
    print(str(round(persent * 100 / len(mappingFiles), 1)) + '%', end='')
    print('\r', end='')

    # fileIn = open("./mapping/" + file, "r")
    fileIn = open("./pred/test/" + file, "r")

    mapping = fileIn.readline()
    mappingList.append(int(mapping[:-1]))

    fileIn.close()

mappingVec = np.array(mappingList)
# numClass = np.max(mappingVec) + 1
numClass = 7
mappingVec = np_utils.to_categorical(mappingVec, numClass)

# sys.exit(0)

# print(mappingVec[0])
# mappingVec = normalize(mappingVec, 1)
# print(mappingVec[0])

print('> Preparing for train...')
lenMapStr = 4
numFilt = 16
lam = 0.0001
# kernel_regularizer=l2(lam)

convSize = 3
paddSize = 1
model = Sequential()

if (rank == 0):
    model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                     activation='relu', input_shape=(matrixDim, matrixDim, 1)))
    model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                     activation='relu'))
    model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                     activation='relu'))
    model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    for i in range(len(model.layers)):
        weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1)) + ".npy")
        bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1)) + ".npy")
        if len(weig) > 0:
            model.get_layer(index=(i + 1)).set_weights([weig, bias])

if (rank == 1):
    model.add(Dense(200, activation='relu', input_shape=(matrixDim * numFilt,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(numClass, activation='softmax'))
    for i in range(len(model.layers)):
        weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1 + 10)) + ".npy")
        bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1 + 10)) + ".npy")
        model.get_layer(index=(i + 1)).set_weights([weig, bias])

persent = -1
print('> Predict on test data...')
t1 = mpi.Wtime()
for i in range(len(matrixVec)):
    persent += 1
    print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
    print('\r', end='')

    if (rank == 0):
        pred = model.predict(matrixVec[i:i + 1])
        comm.send(pred, dest=1, tag=0)
        # print(rank, 'send')
    if (rank == 1):
        tmp = comm.recv(source=0, tag=0)
        # print(rank, 'recv')
        pred = model.predict(tmp)
        # print(rank, 'predict')
        # print("./pred/prediction/mapping" + str(i + 1) + "Pred")
        # print(pred)
        fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

        fileOut.write(str(np.where(pred == pred.max())[1][0]))
        fileOut.write('\n')
        # print(str(np.where(pred == pred.max())))
        # print(str(np.where(pred == pred.max())[1][0]))
        fileOut.close()
t2 = mpi.Wtime() - t1
print('Time =', t2)
# if (rank == 1):
#     score = model.evaluate(matrixVec, mappingVec, batch_size=50)
#     print(score)


# import os
# import numpy as np

# from keras.models import Sequential, load_model
# from keras.utils import plot_model, np_utils


# model = load_model('/mnt/f/prog/magadisser/neuralKeras/nets/goodNet3.h5')
# model.summary()
# model.get_layer(index=2).get_weights()

# fileOut = open("/mnt/f/prog/magadisser/neuralKeras/layers" + str(i + 1), "w")


# for i in range(len(model.layers)):
#     # print(model.get_layer(index=i).get_weights())
#     if len(model.get_layer(index=(i + 1)).get_weights()) > 0:
#         np.save("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str(i + 1), np.array(model.get_layer(index=(i + 1)).get_weights()[0]))
#         np.save("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str(i + 1), np.array(model.get_layer(index=(i + 1)).get_weights()[1]))
#     else:
#         np.save("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str(i + 1), np.array(model.get_layer(index=(i + 1)).get_weights()))
#         np.save("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str(i + 1), np.array(model.get_layer(index=(i + 1)).get_weights()))

# a = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer1.npy")


# np.save("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str(1), np.array(model.get_layer(index=1).get_weights()))

