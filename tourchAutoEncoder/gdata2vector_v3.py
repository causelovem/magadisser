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

    strucFeat = structure.x.numpy()
    strucShape = structure.x.shape[0]
    # считаем среднее по самому графу белка
    strucAvg = strucFeat.sum(axis=0) / len(strucFeat)
    np.save(os.path.join(cfg.rawVectorDir, file), np.append(strucAvg, np.float32(strucShape)))

    structure = structure.to(device)
    # используем encoder
    pred = model.encoder(structure.x, structure.edge_index).to('cpu').detach().numpy()

    # считаем среднее по закодированному графу белка
    predAvg = pred.sum(axis=0) / len(pred)
    np.save(os.path.join(cfg.vectorDir, file), np.append(predAvg, np.float32(strucShape)))

    # считаем взвешенное (attention) среднее по закодированному графу белка
    dist = np.linalg.norm(pred - predAvg, axis=1)
    predAttent = pred * ((dist.max() - dist) / (dist.max() - dist.min())).reshape(-1, 1)
    predAttent = predAttent.sum(axis=0) / len(pred)
    np.save(os.path.join(cfg.attentVectorDir, file), np.append(predAttent, np.float32(strucShape)))
