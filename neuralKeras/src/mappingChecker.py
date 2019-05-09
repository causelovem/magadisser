import os

mappingFiles = os.listdir("./pred/prediction")
mappingFiles.sort(key=lambda x: int(x[7:-4]))
# mappingFiles = os.listdir("../mapping/")

fileNumber = 1
for file in mappingFiles:
    fileIn = open("./pred/prediction/" + file, "r")
    # fileIn = open("../mapping/" + file, "r")

    mapping = fileIn.readlines()
    for i in range(len(mapping) - 1):
        mapping[i] = ' '.join(mapping[i].split('\n')[0].split())

    err = ''
    for i in range(len(mapping) - 1):
        for j in range(i + 1, len(mapping)):
            if (mapping[i] == mapping[j]):
                err = 'In map file {}: str {} and {} are equal to "{}"'.format(
                    file, i + 1, j + 1, mapping[i])
                print(err)
                break
        if (err != ''):
            break

    if (err == ''):
        msg = 'Map file {}: OK!'.format(fileNumber)
        print(msg)

    fileIn.close()
    fileNumber += 1
