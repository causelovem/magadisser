import os

mappingFiles = os.listdir("./pred/prediction")
mappingFiles.sort(key=lambda x: int(x[7:-4]))

mappingFilesTrue = os.listdir("./pred/test")
mappingFilesTrue.sort(key=lambda x: int(x[7:]))
# mappingFiles = os.listdir("../mapping/")

ok = 0
error = 0
for i in range(len(mappingFiles)):
    fileIn = open("./pred/prediction/" + mappingFiles[i], "r")
    fileInTrue = open("./pred/test/" + mappingFilesTrue[i], "r")
    # fileIn = open("../mapping/" + file, "r")

    mapping = int(fileIn.readline()[:-1])
    mappingTrue = int(fileInTrue.readline()[:-1])

    if (mapping == mappingTrue):
        msg = '> Map file {}: OK!'.format(mappingFiles[i])
        print(msg)
        ok += 1
    else:
        err = '> Map file {}: ERROR!'.format(mappingFiles[i])
        print(err)
        error += 1

    print(mapping, mappingTrue)

    fileIn.close()

print('')
res = '> {} OKs and {} ERRORs out of {} files'.format(ok, error, len(mappingFiles))
print(res)
