# основной файл обучения

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as f
import random
import os
import torch_geometric
from gaeModel import AutoEncoder
import config as cfg
import pdb2Gdata_v4 as p2d
from tqdm import tqdm


# фиксируем сиды для воспреизведения
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# класс для считывания данных из папки
class readData(torch_geometric.data.Dataset):
    def __init__(self, fileDir, files):
        self.fileDir = fileDir
        self.files = files
        self.threshold = 7
        self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if cfg.pdbFile:
            return p2d.pdb2Gdata(self.fileDir, self.files[index])
        else:
            return torch.load(os.path.join(self.fileDir, self.files[index]))


set_seed(23)

fileDir = cfg.fileDir
dataList = os.listdir(fileDir)
# dataList = dataList[:5000]

# генерим обучающую и валидационную выборки
validateLength = int(len(dataList) * cfg.validatePart)
dataSizes = [len(dataList) - validateLength, validateLength]
dataTrainRaw, dataValidateRaw = torch.utils.data.random_split(dataList, dataSizes)

# считываем данные
dataTrain = readData(fileDir, dataTrainRaw)
dataValidate = readData(fileDir, dataValidateRaw)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(cfg.device)

# создаем специальные датасеты для параллельной вычитки, формировая батча, перемешивания и тд
trainLoader = torch_geometric.data.DataLoader(dataTrain, batch_size=cfg.batchSize,
                                              num_workers=cfg.numWorkers, shuffle=True)
validateLoader = torch_geometric.data.DataLoader(dataValidate, batch_size=cfg.batchSize,
                                                 num_workers=cfg.numWorkers, shuffle=True)

numOfTrainButch = len(trainLoader)
numOfValidButch = len(validateLoader)

# создаем модель
model = AutoEncoder()
# print(model)
lossType = nn.MSELoss()
# lossType = f.nll_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.000001)

# checkpoint = torch.load(os.path.join(cfg.modelsDir, 'modelCheckpoint{}.pt'.format(epoch)))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model = model.to(device)

# проходимся количество эпох раз
for epoch in range(cfg.epochsNum):
    print('Epoch {}/{}:'.format(epoch, cfg.epochsNum - 1), flush=True)

    # обучаем модель
    sumLoss = 0
    model.train()
    for data in tqdm(trainLoader):
        data = data.to(device)
        preds = model(data).to(device)

        loss = lossType(preds, data.x)
        # считаем ошибку
        sumLoss += float(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print('Train Loss: {:.4f}'.format(float(loss)), flush=True)
    # усредняем ошибку по батчам
    print('Train Loss: {:.4f}'.format(float(sumLoss) / numOfTrainButch), flush=True)

    # сохраняем промежуток для бэкапа
    # потом можно будет использовать эту эпоху в виде обученной модели
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(cfg.modelsDir, 'modelCheckpoint{}.pt'.format(epoch)))
    print('Checkpoint saved to modelCheckpoint{}.pt'.format(epoch))

    # проверяем на валидационной выборке
    sumLoss = 0
    model.eval()
    for data in tqdm(validateLoader):
        data = data.to(device)
        preds = model(data).to(device)

        loss = lossType(preds, data.x)
        sumLoss += float(loss)
    # print('Validation Loss: {:.4f}'.format(float(loss)), flush=True)
    print('Validation Loss: {:.4f}'.format(float(sumLoss) / numOfValidButch), flush=True)

torch.save(model.state_dict(), os.path.join(cfg.modelsDir, 'testModel.pt'))
