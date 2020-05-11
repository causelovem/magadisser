import torch
import os
from gaeModel import AutoEncoder
import numpy as np
import config as cfg
from tqdm import tqdm

# подготавливаем модель
device = cfg.device
model = AutoEncoder()
model.load_state_dict(torch.load(os.path.join(cfg.modelsDir, 'testModel3.pt')))
model = model.to(device)
model.eval()

# названием файлов для обработки
dataList = os.listdir(cfg.fileDir)

for file in tqdm(dataList):
    # загружаем файл
    structure = torch.load(os.path.join(cfg.fileDir, file))
    structure = structure.to(device)
    # используем encoder
    pred = model.encoder(structure.x, structure.edge_index).to('cpu').detach().numpy()

    strucFeat = structure.x.to('cpu').numpy()
    # считаем среднее по самому графу белка
    strucAvg = strucFeat.sum(axis=0) / len(strucFeat)
    np.save(os.path.join(cfg.rawVectorDir, file), strucAvg)

    # считаем среднее по закодированному графу белка
    predAvg = pred.sum(axis=0) / len(pred)
    np.save(os.path.join(cfg.vectorDir, file), predAvg)

    # считаем взвешенное (attention) среднее по закодированному графу белка
    dist = np.linalg.norm(pred - predAvg, axis=1)
    predAttent = pred * ((dist.max() - dist) / (dist.max() - dist.min())).reshape(-1, 1)
    # predAttent = pred / (1 + np.exp(-((1 / np.linalg.norm(pred - predAvg, axis=1)).reshape(-1, 1))))
    predAttent = predAttent.sum(axis=0) / len(pred)
    np.save(os.path.join(cfg.attentVectorDir, file), predAttent)
