# import setuptools
import os
# import sys
import numpy as np

# import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD

from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
# basic class for specifying and training a neural network
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, merge
# utilities for one-hot encoding of ground truth values
from keras.utils import np_utils
from keras.regularizers import l2  # L2-regularisation
from keras.layers.normalization import BatchNormalization  # batch normalisation
from keras.preprocessing.image import ImageDataGenerator  # data augmentation
from keras.callbacks import EarlyStopping  # early stopping

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

print('> Readind matrix data...')
for file in matrixFiles:
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

    tmp = np.array(matrix)
    tmp = np.expand_dims(tmp, axis=2)
    matrixList.append(tmp)

    fileIn.close()

matrixVec = np.array(matrixList)
matrixDim = int(matrixVec.shape[1])
numOfSet = int(matrixVec.shape[0])

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
for file in mappingFiles:
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
            mapping[i][j] = int(mapping[i][j]) / 10
    tmp = np.array(mapping)
    mappingList.append(tmp)

    fileIn.close()

mappingVec = np.array(mappingList)
mappingVec = mappingVec.reshape(numOfSet, matrixDim * 4)

print('> Preparing for train...')
lenMapStr = 4
numFilt = 16

convSize = 3
paddSize = 1

X_val = matrixVec[70:]
Y_val = mappingVec[70:]
X_train = matrixVec[:70]
Y_train = mappingVec[:70]


batch_size = 10  # in each iteration, we consider 128 training examples at once
num_epochs = 50  # we iterate at most fifty times over the entire training set
kernel_size = 3  # we will use 3x3 kernels throughout
pool_size = 2  # we will use 2x2 pooling throughout
conv_depth = 32  # use 32 kernels in both convolutional layers
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
# there will be 128 neurons in both hidden layers
hidden_size = int(matrixDim * matrixDim * numFilt / (32 * 32))
l2_lambda = 0.0001  # use 0.0001 as a L2-regularisation factor
ens_models = 3

inp = Input(shape=(matrixDim, matrixDim, 1))
# Apply BN to the input (N.B. need to rename here)
inp_norm = BatchNormalization(axis=1)(inp)

outs = []  # the list of ensemble outputs
for i in range(ens_models):
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer), applying BN in between
    conv_1 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                    kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(inp_norm)
    conv_1 = BatchNormalization(axis=1)(conv_1)
    conv_2 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                    kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(conv_1)
    conv_2 = BatchNormalization(axis=1)(conv_2)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    conv_11 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                     kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(pool_1)
    conv_11 = BatchNormalization(axis=1)(conv_11)
    conv_22 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                     kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(conv_11)
    conv_22 = BatchNormalization(axis=1)(conv_22)
    pool_11 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_22)
    drop_11 = Dropout(drop_prob_1)(pool_1)

    conv_111 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                      kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(pool_11)
    conv_111 = BatchNormalization(axis=1)(conv_111)
    conv_222 = Conv2D(numFilt, (convSize, convSize), border_mode='same',
                      kernel_initializer='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(conv_111)
    conv_222 = BatchNormalization(axis=1)(conv_222)
    pool_111 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_222)
    drop_111 = Dropout(drop_prob_1)(pool_111)

    flat = Flatten()(drop_111)
    hidden = Dense(hidden_size, kernel_initializer='he_uniform', W_regularizer=l2(
        l2_lambda), activation='relu')(flat)  # Hidden ReLU layer
    hidden = BatchNormalization(axis=1)(hidden)
    drop = Dropout(drop_prob_2)(hidden)
    outs.append(Dense(matrixDim * lenMapStr, kernel_initializer='glorot_uniform', W_regularizer=l2(
        l2_lambda), activation='softmax')(drop))  # Output softmax layer

# average the predictions to obtain the final output
out = merge(outs, mode='ave')

# To define a model, just specify its input and output layers
model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy

datagen = ImageDataGenerator(
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)
datagen.fit(X_train)

# fit the model on the batches generated by datagen.flow()---most parameters similar to model.fit
model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=num_epochs,
                    validation_data=(X_val, Y_val),
                    verbose=1)

# Evaluate the trained model on the test set!
model.evaluate(X_train, Y_train, verbose=1)


model.save('./nets/net1.h5')


print('> Predict on trannig data...')
for i in range(len(matrixVec)):
    pred = model.predict(matrixVec[i:i + 1])
    print("./prediction/mapping" + str(i + 1) + "Pred")
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
                tmp = int(round(pred[0][j * lenMapStr + k] * 10))
                if (tmp > max - 1):
                    fileOut.write(str(int(0)) + ' ')
                else:
                    fileOut.write(str(int(tmp)) + ' ')
            else:
                fileOut.write(str(int(0)) + ' ')
        fileOut.write('\n')

    fileOut.close()

# print(score)
