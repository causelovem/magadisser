#!/usr/bin/python3
# import setuptools
import os
# import sys
import numpy as np

from keras.applications.vgg19 import VGG19
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model, normalize, np_utils
# import matplotlib.pyplot as plt
# from keras.regularizers import l2


from mpi4py import rc
rc.initialize = False

from mpi4py import MPI as mpi
import queue as qu
from time import sleep

# np.set_printoptions(threshold=np.nan)

mpi.Init()
comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if (rank == 0):
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
    # numOfSet = int(matrixVec.shape[0])

    # print(matrixVec[0])
    # matrixVec = normalize(matrixVec, 2)
    # print(matrixVec[0])

    print(matrixVec.shape)

# sys.exit(0)

# mappingList = []
# # mappingFiles = os.listdir("./mapping")
# mappingFiles = os.listdir("./pred/test")
# mappingFiles.sort(key=lambda x: int(x[7:]))

# persent = -1
# print('> Readind mapping data...')
# for file in mappingFiles:
#     persent += 1
#     print(str(round(persent * 100 / len(mappingFiles), 1)) + '%', end='')
#     print('\r', end='')

#     # fileIn = open("./mapping/" + file, "r")
#     fileIn = open("./pred/test/" + file, "r")

#     mapping = fileIn.readline()
#     mappingList.append(int(mapping[:-1]))

#     fileIn.close()

# mappingVec = np.array(mappingList)
# # numClass = np.max(mappingVec) + 1
# numClass = 7
# mappingVec = np_utils.to_categorical(mappingVec, numClass)

# sys.exit(0)

# print(mappingVec[0])
# mappingVec = normalize(mappingVec, 1)
# print(mappingVec[0])
if (rank != 0):
    matrixVec = []
lenMatrixVec = comm.bcast(len(matrixVec), root=0)

print('> Preparing for train...')
lenMapStr = 4
numFilt = 16

convSize = 3
paddSize = 1
# model = Sequential()
# model = load_model('/mnt/f/prog/magadisser/neuralKeras/nets/goodNet3.h5')

if 1 == 1:
    img_input = Input(shape=[256, 256, 1])
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv4')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv5')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv4')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv5')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv5')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv5')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv5')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(4096, activation='relu', name='fc3')(x)
    x = Dense(4096, activation='relu', name='fc4')(x)
    x = Dense(7, activation='softmax', name='predictions')(x)

    model = Model(inputs=img_input, outputs=x)
else:
    model = Sequential()
    # model.add(Input(shape=[256, 256, 1]))
    model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                     activation='relu', input_shape=(256, 256, 1)))
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

    # for i in range(len(model.layers)):
    #     weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1)) + ".npy")
    #     bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1)) + ".npy")
    #     if len(weig) > 0:
    #         model.get_layer(index=(i + 1)).set_weights([weig, bias])

    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # for i in range(len(model.layers)):
    #     weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1 + 10)) + ".npy")
    #     bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1 + 10)) + ".npy")
    #     model.get_layer(index=(i + 1)).set_weights([weig, bias])


modelLen = len(model.layers)
allPos = []

# divade by layers
if 1 == 0:
    nearCount = model.count_params() // size
    numOfLayers = modelLen // size
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
else:
# divade by layers' wieghts
    nearCount = model.count_params() // size
    s = 0
    cnt = 0
    allPos = []
    for i in range(modelLen):
        s += model.get_layer(index=i).count_params()
        cnt += 1
        if (s >= nearCount) or (i == modelLen - 1) or (modelLen - i <= size - len(allPos)):
            allPos.append(cnt)
            s = 0
            cnt = 0
    # print(allPos)
    startLayer = sum(allPos[0:rank])
    # print(rank, startLayer)

    for i in range(startLayer, sum(allPos[0:rank + 1])):
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
print(rank, nearCount, modelDiv.count_params())

comm.Barrier()

# for i in range(startLayer, startLayer + numOfLayers):
#     weig = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/layer" + str((i + 1)) + ".npy")
#     bias = np.load("/mnt/f/prog/magadisser/neuralKeras/layers/biase" + str((i + 1)) + ".npy")
#     if len(weig) > 0:
#         model.get_layer(index=(i + 1)).set_weights([weig, bias])

que = qu.Queue()


prevRank = rank - 1
nextRank = rank + 1

persent = -1
print('> Predict on test data...')
step = 5
que = []
t1 = mpi.Wtime()
if 1 == 0:
    for i in range(lenMatrixVec):
        persent += 1
        print(str(round(persent * 100 / lenMatrixVec, 1)) + '%', end='')
        print('\r', end='')

        if (rank == 0):
            req = comm.send(modelDiv.predict(matrixVec[i:i + 1]), dest=1, tag=0)
        elif (rank != size - 1):
            pred = modelDiv.predict(comm.recv(source=prevRank, tag=0))
            comm.send(pred, dest=nextRank, tag=0)
        elif (rank == size - 1):
            pred = modelDiv.predict(comm.recv(source=prevRank, tag=0))
            # print("./pred/prediction/mapping" + str(i + 1) + "Pred")
            # fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

            # fileOut.write(str(np.where(pred == pred.max())[1][0]))
            # fileOut.write('\n')
            # # print(str(np.where(pred == pred.max())))
            # # print(str(np.where(pred == pred.max())[1][0]))
            # # fileOut.close()

if 1 == 1:
    processed = 0
    if (rank == 0):
        for i in range(lenMatrixVec):
            persent += 1
            print(str(round(persent * 100 / lenMatrixVec, 1)) + '%', end='')
            print('\r', end='')

            que.append(modelDiv.predict(matrixVec[i:i + 1]))
            if (i % step == 0) or (i == lenMatrixVec - 1):
                req = comm.send(que, dest=1, tag=0)
                que = []
    elif (rank != size - 1):
        while 1:
            data = comm.recv(source=prevRank, tag=0)
            processed += len(data)
            for el in data:
                que.append(modelDiv.predict(el))
            comm.send(que, dest=nextRank, tag=0)
            if (processed == lenMatrixVec):
                break
            que = []
    elif (rank == size - 1):
        while 1:
            data = comm.recv(source=prevRank, tag=0)
            processed += len(data)
            for el in data:
                pred = modelDiv.predict(el)
            if (processed == lenMatrixVec):
                break
t2 = mpi.Wtime() - t1

t2 = np.array(t2)
resTime = np.array(0.0)
comm.Reduce(t2, resTime, op=mpi.MAX, root=0)
if (rank == 0):
    print('> Time =', resTime)

print(rank, 'END')
comm.Barrier()

mpi.Finalize()



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
