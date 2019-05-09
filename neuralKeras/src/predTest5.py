# import setuptools
import os
# import sys
import numpy as np

from keras.models import Sequential, load_model
from keras.utils import plot_model, np_utils
import time

np.set_printoptions(threshold=np.nan)

matrixList = []
matrixFiles = os.listdir("./pred/matrix")
matrixFiles.sort(key=lambda x: int(x[6:]))

persent = -1
weig = [4.04, 0.78, 0.46, 0.42, 0.41, 0.32]
print('> Readind matrix data...')
for file in matrixFiles:
    persent += 1
    print(str(round(persent * 100 / len(matrixFiles), 1)) + '%', end='')
    print('\r', end='')

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

persent = -1
print('> Readind mapping data...')
for file in mappingFiles:
    persent += 1
    print(str(round(persent * 100 / len(mappingFiles), 1)) + '%', end='')
    print('\r', end='')

    fileIn = open("./pred/test/" + file, "r")

    mapping = fileIn.readline()
    mappingList.append(int(mapping[:-1]))

    fileIn.close()

mappingVec = np.array(mappingList)
# numClass = np.max(mappingVec) + 1
numClass = 7
mappingVec = np_utils.to_categorical(mappingVec, numClass)
# mappingVec = mappingVec.reshape(numOfSet, matrixDim * 4)

# sys.exit(0)

print('> Preparing for prediction...')
lenMapStr = 4
model = Sequential()
# model = load_model('./nets/net1.h5')
# model = load_model('/mnt/f/prog/magadisser/neuralKeras/nets/goodNet3.h5')
model = load_model('./nets/goodNet3.h5')
# plot_model(model, to_file='model.png')

persent = -1
print('> Predict on test data...')
t1 = time.time()
for i in range(len(matrixVec)):
    persent += 1
    print(str(round(persent * 100 / len(matrixVec), 1)) + '%', end='')
    print('\r', end='')

    pred = model.predict(matrixVec[i:i + 1])
    # print("./pred/prediction/mapping" + str(i + 1) + "Pred")
    # print(pred)
    fileOut = open("./pred/prediction/mapping" + str(i + 1) + "Pred", "w")

    fileOut.write(str(np.where(pred == pred.max())[1][0]))
    fileOut.write('\n')
    # print(str(np.where(pred == pred.max())))
    # print(str(np.where(pred == pred.max())[1][0]))

    fileOut.close()
t2 = time.time() - t1
print('Time =', t2)
# score = model.evaluate(matrixVec, mappingVec, batch_size=50)
# print(score)
