import os
from tqdm import tqdm
import config as cfg
import pdb2Gdata_v2 as p2d


dataList = os.listdir(cfg.rawFileDir)

currDir = os.getcwd()
myN = int(currDir[-1:])
print(myN)

part = len(dataList) // 10

if myN == 9:
    print(part * myN, len(dataList))
    dataList = dataList[part * myN:]
else:
    print(part * myN, part * (myN + 1))
    dataList = dataList[part * myN:part * (myN + 1)]

errors = open('errors.txt', 'w')
for file in tqdm(dataList):
    try:
        p2d.pdb2Gdata(cfg.rawFileDir, file, cfg.fileDir)
    except Exception as e:
        errors.write(str(e) + '\n')
        errors.write(file + '\n\n')

errors.close()
