# import setuptools
import os
import sys
import numpy as np
import math

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils import plot_model, normalize, np_utils
import matplotlib.pyplot as plt
from keras.regularizers import l2

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

persent = -1
# quan = 16
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

    # for i in range(dim):
    #     size = dim
    #     while size > 1:
    #         size //= 2
    #         tmp = []
    #         for j in range(size):
    #             matrix[i][j] = (matrix[i][2 * j] + matrix[i][2 * j + 1]) / 2
    #             tmp.append(matrix[i][j] - matrix[i][2 * j + 1])
    #         for j in range(len(tmp)):
    #             matrix[i][j + size] = tmp[j]

    # for i in range(dim):
    #     size = dim
    #     while size > 1:
    #         size //= 2
    #         tmp = []
    #         for j in range(size):
    #             matrix[j][i] = (matrix[2 * j][i] + matrix[2 * j + 1][i]) / 2
    #             tmp.append(matrix[j][i] - matrix[2 * j + 1][i])
    #         for j in range(len(tmp)):
    #             matrix[j + size][i] = tmp[j]

    # for i in range(dim):
    #     for j in range(dim):
    #         matrix[i][j] *= weig[int(min(max(math.log(i + 1, 2), math.log(j + 1, 2)), 5))]

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

    tmp = np.array(matrix)
    # tmp = np.array(matrix)
    # print(matrix)
    # print('\r\n')
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

print('> Readind mapping data...')
# for file in mappingFiles:
#     fileIn = open("./mapping/" + file, "r")

#     mapping = fileIn.readlines()
#     dim = len(mapping)

#     # for i in range(dim):
#     #     mapping[i] = mapping[i][:-1].split(' ')
#     #     strDim = len(mapping[i])
#     #     for j in range(strDim):
#     #         if (mapping[i][j] != '0'):
#     #             mapping[i][j] = 1.0 / int(mapping[i][j])
#     #         else:
#     #             mapping[i][j] = int(mapping[i][j])
#     # tmp = np.array(mapping)
#     # mappingList.append(tmp)

#     for i in range(dim):
#         mapping[i] = mapping[i][:-1].split(' ')
#         strDim = len(mapping[i])
#         for j in range(strDim):
#             # mapping[i][j] = int(mapping[i][j]) * step
#             # mapping[i][j] = int(mapping[i][j]) / 10
#             mapping[i][j] = int(mapping[i][j]) / max
#             # mapping[i][j] = int(mapping[i][j])
#     tmp = np.array(mapping)
#     mappingList.append(tmp)

#     fileIn.close()

for file in mappingFiles:
    fileIn = open("./mapping/" + file, "r")

    mapping = fileIn.readline()
    # print(mapping[:-1])
    mappingList.append(int(mapping[:-1]))

    fileIn.close()

mappingVec = np.array(mappingList)
# print(mappingVec.shape)
# print(mappingVec[0])
numClass = np.max(mappingVec) + 1
# print(numClass)

mappingVec = np_utils.to_categorical(mappingVec, numClass)
# print(mappingVec.shape)
# print(mappingVec[0])
# sys.exit(0)

# mappingVec = mappingVec.reshape(numOfSet, matrixDim * 4)

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

# input_shape=(matrixDim, matrixDim, 1)

# model.add(Input(shape=(matrixDim, matrixDim, 1)))
# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3, input_shape=(matrixDim, matrixDim, 1)))
model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
# # model.add(ZeroPadding2D((paddSize, paddSize)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
# model.add(Dense(int(matrixDim * matrixDim * numFilt * 4 / (32 * 32)),
#                 activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(int(matrixDim * matrixDim * numFilt / (32 * 32)),
#                 activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(matrixDim * lenMapStr,
#                 activation='relu'))
# model.add(Dropout(0.5))

model.add(BatchNormalization(axis=1))
model.add(Dense(200,
                activation='relu'))
# model.add(Dropout(0.5))

model.add(BatchNormalization(axis=1))
model.add(Dense(200,
                activation='relu'))
# model.add(Dropout(0.5))

model.add(BatchNormalization(axis=1))
model.add(Dense(200,
                activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(matrixDim * lenMapStr,
#                 activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(matrixDim * lenMapStr, activation='sigmoid'))
model.add(Dense(numClass, activation='softmax'))

# model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
#                  kernel_initializer='he_uniform', activation='relu'))
# kernel_initializer='glorot_uniform' 'he_uniform'
# model = load_model('./nets/net2.h5')

# softplus softsign softmax relu sigmoid/hard_sigmoid

# Adadelta Adam sgd
# poisson mse logcosh mean_squared_logarithmic_error categorical_hinge
# sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

xVal = matrixVec[int(numOfSet * 0.75):]
yVal = mappingVec[int(numOfSet * 0.75):]

xTrain = matrixVec[:int(numOfSet * 0.75)]
yTrain = mappingVec[:int(numOfSet * 0.75)]

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# history = model.fit(matrixVec, mappingVec, epochs=50, batch_size=50,
#                     callbacks=[EarlyStopping(monitor='loss', patience=5)])
# score = model.evaluate(matrixVec, mappingVec, batch_size=50)
history = model.fit(xTrain, yTrain, epochs=30, batch_size=50,
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
# plt.legend(['train', 'test'], loc='upper left')
plt.legend(['train', 'test'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('plot.png', fmt='png')
# plt.show()
plt.clf()


persent = -1
print('> Predict on trannig data...')
# for i in range(len(matrixVec)):
#     persent += 1
#     print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
#     print('\r', end='')

#     pred = model.predict(matrixVec[i:i + 1])
#     # print("./prediction/mapping" + str(i + 1) + "Pred")
#     # print(pred)
#     fileOut = open("./prediction/mapping" + str(i + 1) + "Pred", "w")

#     # for j in range(matrixDim):
#     #     for k in range(lenMapStr):
#     #         if (abs(pred[0][j * lenMapStr + k] - 0) > 0.001):
#     #             tmp = 1 / pred[0][j * lenMapStr + k]
#     #             if (round(tmp) > max - 1):
#     #                 fileOut.write(str(int(0)) + ' ')
#     #             else:
#     #                 fileOut.write(str(int(round(tmp))) + ' ')
#     #         else:
#     #             fileOut.write(str(int(0)) + ' ')
#     #     fileOut.write('\n')

#     # fileOut.close()

#     for j in range(matrixDim):
#         for k in range(lenMapStr):
#             if (abs(pred[0][j * lenMapStr + k] - 0) > 0.001):
#                 # tmp = int(round(pred[0][j * lenMapStr + k] / step))
#                 # tmp = int(round(pred[0][j * lenMapStr + k] * 10))
#                 tmp = int(round(pred[0][j * lenMapStr + k] * max))
#                 # tmp = int(round(pred[0][j * lenMapStr + k]))
#                 if (tmp > max - 1):
#                     fileOut.write(str(int(0)) + ' ')
#                 else:
#                     fileOut.write(str(int(tmp)) + ' ')
#             else:
#                 fileOut.write(str(int(0)) + ' ')
#         fileOut.write('\n')

#     fileOut.close()
for i in range(len(matrixVec)):
    persent += 1
    print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
    print('\r', end='')

    pred = model.predict(matrixVec[i:i + 1])
    # print("./prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./prediction/mapping" + str(i + 1) + "Pred", "w")

    fileOut.write(str(np.where(pred == pred.max())[1][0]))
    # print(str(np.where(pred == pred.max())))
    # print(str(np.where(pred == pred.max())[1][0]))

    # for j in range(matrixDim):
    #     for k in range(lenMapStr):
    #         if (abs(pred[0][j * lenMapStr + k] - 0) > 0.001):
    #             # tmp = int(round(pred[0][j * lenMapStr + k] / step))
    #             # tmp = int(round(pred[0][j * lenMapStr + k] * 10))
    #             tmp = int(round(pred[0][j * lenMapStr + k] * max))
    #             # tmp = int(round(pred[0][j * lenMapStr + k]))
    #             if (tmp > max - 1):
    #                 fileOut.write(str(int(0)) + ' ')
    #             else:
    #                 fileOut.write(str(int(tmp)) + ' ')
    #         else:
    #             fileOut.write(str(int(0)) + ' ')
    #     fileOut.write('\n')

    fileOut.close()

print(score)
