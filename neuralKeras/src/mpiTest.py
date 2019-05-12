# import setuptools
import os
# import sys
import numpy as np

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model, normalize, np_utils
# import matplotlib.pyplot as plt
# from keras.regularizers import l2

from mpi4py import MPI as mpi
# import queue as qu

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

convSize = 3
paddSize = 1
# model = Sequential()
model = load_model('/mnt/f/prog/magadisser/neuralKeras/nets/goodNet3.h5')
modelLen = len(model.layers)

numOfLayers = modelLen // size
allPos = []
for i in range(size):
    allPos.append(numOfLayers)
    if (i < modelLen % size):
        allPos[i] += 1
if (rank < modelLen % size):
    numOfLayers += 1

startLayer = sum(allPos[0:rank])
# print(rank, startLayer, numOfLayers, allPos)

for i in range(startLayer, startLayer + numOfLayers):
    if (i == 0):
        # inp = Input(shape=(matrixDim, matrixDim, 1))
        inp = Input([d.value for d in model.layers[i].get_input_at(0).shape[1:]])
        lay = model.layers[i](inp)
    elif (i == startLayer):
        inp = Input(shape=[d.value if d.value is not None else np.prod(model.layers[i - 1].get_input_at(0).shape[1:]).value for d in model.layers[i - 1].get_output_at(0).shape[1:]])
        lay = model.layers[i](inp)
    else:
        lay = model.layers[i](lay)

modelDiv = Model(inputs=inp, outputs=lay)
# modelDiv.summary()

comm.Barrier()

# for i in range(startLayer, startLayer + numOfLayers):
#     weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1)) + ".npy")
#     bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1)) + ".npy")
#     if len(weig) > 0:
#         model.get_layer(index=(i + 1)).set_weights([weig, bias])

# que = qu.Queue()

if 0 == 1:
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
lenMatrixVec = len(matrixVec)
for i in range(lenMatrixVec):
    persent += 1
    print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
    print('\r', end='')

    if (rank == 0):
        # pred = modelDiv.predict(matrixVec[i:i + 1])
        # comm.send(pred, dest=1, tag=0)
        # req = comm.isend(modelDiv.predict(matrixVec[i:i + 1]), dest=1, tag=0)
        comm.send(modelDiv.predict(matrixVec[i:i + 1]), dest=1, tag=0)

        # if (i == lenMatrixVec - 1):
        #     req.wait()
        #     print(rank, 'send')
    elif (rank != size - 1):
        pred = modelDiv.predict(comm.recv(source=(rank - 1), tag=0))
        comm.send(pred, dest=(rank + 1), tag=0)
    elif (rank == size - 1):
        # tmp = comm.recv(source=0, tag=0)
        # pred = modelDiv.predict(tmp)

        pred = modelDiv.predict(comm.recv(source=(rank - 1), tag=0))
        # print("./pred/prediction/mapping" + str(i + 1) + "Pred")
        # fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

        # fileOut.write(str(np.where(pred == pred.max())[1][0]))
        # fileOut.write('\n')
        # # print(str(np.where(pred == pred.max())))
        # # print(str(np.where(pred == pred.max())[1][0]))
        # fileOut.close()
t2 = mpi.Wtime() - t1

t2 = np.array(t2)
resTime = np.array(0.0)
comm.Reduce(t2, resTime, op=mpi.MAX, root=0)
if (rank == 0):
    print('> Time =', resTime)

print(rank, 'END')
comm.Barrier()

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
