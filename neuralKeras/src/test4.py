# import setuptools
import os
import sys
import numpy as np
import math

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt


def checkHowMuch(vec):
    size = len(vec)
    for i in range(size):
        if (vec[i] == -1):
            return i

    return -1


def checkDuplicate(vec, pos):
    size = len(vec)
    for i in range(size):
        if (vec[i] == pos):
            return -1

    return 1


np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

persent = -1
print('> Readind matrix data...')
for file in matrixFiles:
    persent += 1
    print(str(round(persent * 100 / len(matrixFiles), 1)) + '%', end='')
    print('\r', end='')

    fileIn = open("./matrix/" + file, "r")

    matrix = fileIn.readlines()
    dim = len(matrix)
    transMatrix = np.zeros((dim, 6)) - np.ones((dim, 6))

    for i in range(dim):
        matrix[i] = matrix[i][:-2].split(' ')
        # for j in range(len(matrix[i])):
        #     matrix[i][j] = int(matrix[i][j])

        pairList = []
        for j in range(dim):
            pairList.append((j, int(matrix[i][j])))

        pairList.sort(key=lambda x: x[1], reverse=True)
        countNeigh = checkHowMuch(transMatrix[i])
        countNum = 0

        while ((countNeigh < 6) and (countNeigh != -1) and (countNum < dim)):
            if (pairList[countNum][1] == 0):
                break

            pos = pairList[countNum][0]
            place2 = checkHowMuch(transMatrix[pos])
            val1 = checkDuplicate(transMatrix[i], pos)
            val2 = checkDuplicate(transMatrix[pos], i)

            if (val1 == 1):
                if (place2 != -1):
                    if (val2 == 1):
                        transMatrix[i][countNeigh] = pos
                        countNeigh += 1
                        transMatrix[pos][place2] = i
            countNum += 1

    transMatrix += np.ones((dim, 6))

    # tmp = np.array(matrix)
    tmp = transMatrix
    # tmp = np.array(sque)
    # print(tmp)
    # print('\r\n')
    tmp = np.expand_dims(tmp, axis=2)
    matrixList.append(tmp)

    fileIn.close()

matrixVec = np.array(matrixList)
matrixDim = int(matrixVec.shape[1])
numOfSet = int(matrixVec.shape[0])

# matrixVec = matrixVec.reshape(numOfSet, matrixDim * 6)

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

# sys.exit(0)

mappingList = []
mappingFiles = os.listdir("./mapping")
mappingFiles.sort(key=lambda x: int(x[7:]))

step = 1.0 / (max - 1.0)

persent = -1
print('> Readind mapping data...')
for file in mappingFiles:
    persent += 1
    print(str(round(persent * 100 / len(mappingFiles), 1)) + '%', end='')
    print('\r', end='')

    fileIn = open("./mapping/" + file, "r")

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

print('> Preparing for train...')
lenMapStr = 4

model = Sequential()

model.add(Flatten(input_shape=(matrixDim, 6, 1)))
model.add(BatchNormalization(axis=1))
model.add(Dense(int(matrixDim * 2),
                activation='relu'))
# model.add(Dropout(0.3))

model.add(BatchNormalization(axis=1))
model.add(Dense(int(matrixDim * 2),
                activation='relu'))
# model.add(Dropout(0.3))

model.add(BatchNormalization(axis=1))
model.add(Dense(int(matrixDim * 4),
                activation='relu'))
# model.add(Dropout(0.3))

# model.add(BatchNormalization(axis=1))
# model.add(Dense(int(matrixDim * 4),
#                 activation='relu'))
# model.add(Dropout(0.3))

model.add(BatchNormalization(axis=1))
model.add(Dense(matrixDim * lenMapStr,
                activation='sigmoid'))

# kernel_initializer='glorot_uniform' 'he_uniform'
# model = load_model('./nets/net2.h5')

# softplus softsign softmax relu tanh sigmoid/hard_sigmoid

# Adadelta Adam sgd
# poisson mse logcosh mean_squared_logarithmic_error categorical_hinge
# sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
xVal = matrixVec[int(numOfSet * 0.8):]
yVal = mappingVec[int(numOfSet * 0.8):]

xTrain = matrixVec[:int(numOfSet * 0.8)]
yTrain = mappingVec[:int(numOfSet * 0.8)]

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

# history = model.fit(matrixVec, mappingVec, epochs=50, batch_size=50,
#                     callbacks=[EarlyStopping(monitor='loss', patience=5)])
# score = model.evaluate(matrixVec, mappingVec, batch_size=50)
history = model.fit(xTrain, yTrain, epochs=50, batch_size=50,
                    validation_data=(xVal, yVal),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
score = model.evaluate(xTrain, yTrain, batch_size=50)


model.save('./nets/net1.h5')
plot_model(model, to_file='model.png', show_shapes=True)


plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy/loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('plotTest4.png', fmt='png')
# plt.show()
plt.clf()

persent = -1
print('> Predict on trannig data...')
for i in range(len(matrixVec)):
    persent += 1
    print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
    print('\r', end='')

    pred = model.predict(matrixVec[i:i + 1])
    # print("./prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./prediction/mapping" + str(i + 1) + "Pred", "w")

    # for j in range(matrixDim):
    #     for k in range(lenMapStr):
    #         if (abs(pred[0][j * lenMapStr + k] - 0) > 0.001):
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

print(score)
