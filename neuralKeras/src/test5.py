# import setuptools
import os
import sys
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, ZeroPadding2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
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
mappingFiles = os.listdir("./mapping")
mappingFiles.sort(key=lambda x: int(x[7:]))

persent = -1
print('> Readind mapping data...')
for file in mappingFiles:
    persent += 1
    print(str(round(persent * 100 / len(mappingFiles), 1)) + '%', end='')
    print('\r', end='')

    fileIn = open("./mapping/" + file, "r")

    mapping = fileIn.readline()
    mappingList.append(int(mapping[:-1]))

    fileIn.close()

mappingVec = np.array(mappingList)
numClass = np.max(mappingVec) + 1
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

# input_shape=(matrixDim, matrixDim, 1)

# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3, input_shape=(matrixDim, matrixDim, 1)))
model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                 activation='relu', input_shape=(matrixDim, matrixDim, 1)))
# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 2, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
# model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
#                  activation='relu'))
# # model.add(ZeroPadding2D((paddSize, paddSize)))
# # model.add(BatchNormalization(axis=3))
# model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(ZeroPadding2D((paddSize, paddSize)))
# model.add(BatchNormalization(axis=3))
# model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
#                  activation='relu'))
# # # model.add(ZeroPadding2D((paddSize, paddSize)))
# # model.add(BatchNormalization(axis=3))
# model.add(Conv2D(numFilt * 4, (convSize, convSize), padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())

# model.add(BatchNormalization(axis=1))
model.add(Dense(200,
                activation='relu'))
# model.add(Dropout(0.5))

# model.add(BatchNormalization(axis=1))
model.add(Dense(200,
                activation='relu'))
# model.add(Dropout(0.5))

# model.add(BatchNormalization(axis=1))
# model.add(Dense(200,
#                 activation='relu'))
# model.add(Dropout(0.5))

# model.add(BatchNormalization(axis=1))
model.add(Dense(numClass, activation='softmax'))

# model.add(Conv2D(numFilt, (convSize, convSize), padding='same',
#                  kernel_initializer='he_uniform', activation='relu'))
# kernel_initializer='glorot_uniform' 'he_uniform'
# model = load_model('./nets/net2.h5')

# softplus softsign softmax relu sigmoid/hard_sigmoid

# Adadelta Adam sgd
# poisson mse logcosh mean_squared_logarithmic_error categorical_hinge categorical_crossentropy
# sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history = model.fit(matrixVec, mappingVec, epochs=15, batch_size=50,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5),
                               EarlyStopping(monitor='val_acc', patience=10)])
score = model.evaluate(matrixVec, mappingVec, batch_size=50)


model.save('./nets/net1.h5')
plot_model(model, to_file='model.png', show_shapes=True)


plt.subplot(211)
plt.title('Model accuracy/loss (full)')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('Accuracy')
# plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.savefig('plotFull.png', fmt='png')
plt.clf()

plt.subplot(211)
plt.plot(history.history['acc'][2:])
plt.plot(history.history['val_acc'][2:])
plt.title('Model accuracy/loss (part)')
plt.ylabel('Accuracy')
# plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')

plt.subplot(212)
plt.plot(history.history['loss'][2:])
plt.plot(history.history['val_loss'][2:])
# plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper right')
# plt.savefig('plot.png', fmt='png')
plt.savefig('plotPart.png', fmt='png')
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

    fileOut.write(str(np.where(pred == pred.max())[1][0]))
    # print(str(np.where(pred == pred.max())))
    # print(str(np.where(pred == pred.max())[1][0]))

    fileOut.close()

print(score)
