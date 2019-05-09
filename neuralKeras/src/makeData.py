import subprocess
import sys
import time


def command(com):
    result = subprocess.Popen(com.split(), stdout=subprocess.PIPE)
    # result.wait()
    result = result.communicate()[0]
    return result[:-1]


if (len(sys.argv) != 7):
    err = "Unexpected quantity of arguments, check your comand string:\n"
    err += "<matrixDir> <mappingDir> <numOfFiles> <matrixDim> <predDir> <numOfPred>"
    print(err)
    sys.exit(1)

numOfTest = int(sys.argv[3])
matrixDim = int(sys.argv[4])

numOfPred = int(sys.argv[6])

for i in range(numOfTest):
    com = './bin/com_matrix_class_gen {}{}{} {} {}{}{}'.format(
        sys.argv[1], "/matrix", i + 1, matrixDim, sys.argv[2], "/mapping", i + 1)

    print(com)
    command(com)

    # com = './bin/greedy {}{}{} {} '.format(
    #     sys.argv[1], "/matrix", i + 1, matrixDim)
    # com += '{}{}{}'.format(sys.argv[2], "/mapping", i + 1)

    # print(com)
    # command(com)

    time.sleep(1)


for i in range(numOfPred):
    com = './bin/com_matrix_class_gen {}{}{}{} {} {}{}{}{}'.format(
        sys.argv[5], "/matrix", "/matrix", i + 1, matrixDim, sys.argv[5], "/test", "/mapping", i + 1)

    print(com)
    command(com)

    # com = './bin/greedy {}{}{}{} {} '.format(
    #     sys.argv[5], "/matrix", "/matrix", i + 1, matrixDim)
    # com += '{}{}{}{}'.format(sys.argv[5], "/test", "/mapping", i + 1)

    # print(com)
    # command(com)

    time.sleep(1)
