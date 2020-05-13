import os
from tqdm import tqdm
import config as cfg
from pdb2Gdata_v6 import pdb2Gdata


# dataList = os.listdir(cfg.rawFileDir)

# currDir = os.getcwd()
# myN = int(os.path.basename(currDir)[3:])  # dirNNNN
# print(myN)
# totalDir = 12

# part = len(dataList) // totalDir

# if myN == totalDir - 1:
#     print(part * myN, len(dataList))
#     dataList = dataList[part * myN:]
# else:
#     print(part * myN, part * (myN + 1))
#     dataList = dataList[part * myN:part * (myN + 1)]

dataList = os.listdir('/mnt/hdd1/tmp/new')
errors = open('errors.txt', 'w')
for file in tqdm(dataList):
    try:
        pdb2Gdata('/mnt/hdd1/tmp/new', file, '/mnt/hdd1/tmp/newGdata')
        # pdb2Gdata(cfg.rawFileDir, file, cfg.fileDir)
    except Exception as e:
        errors.write(str(e) + '\n')
        errors.write(file + '\n\n')

errors.close()
