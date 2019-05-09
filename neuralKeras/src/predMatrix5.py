import sys
import numpy as np

from keras.models import Sequential, load_model

np.set_printoptions(threshold=np.nan)

# <matrixFile>
if (len(sys.argv) != 2):
    print('> Unexpected quantity of arguments, check your comand string.')
    sys.exit(1)

matrixList = []

fileIn = open(sys.argv[1], "r")

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

print('> Preparing for prediction...')
model = Sequential()
model = load_model('./nets/net1.h5')
# model = load_model('./nets/netnetnet1.h5')

print('> Predict on test data...')
pred = model.predict(matrixVec[0:1])
res = str(np.where(pred == pred.max())[1][0])
resMsg = '> Your matrix class is {}'.format(res)
print(resMsg)
