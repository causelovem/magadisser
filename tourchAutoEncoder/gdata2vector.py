import torch
import os
from gaeModel import AutoEncoder
import numpy as np
import config as cfg
from tqdm import tqdm

device = cfg.device
model = AutoEncoder()
model.load_state_dict(torch.load(os.path.join(cfg.modelsDir, 'testModel.pt')))
model = model.to(device)
model.eval()

dataList = os.listdir(cfg.fileDir)

for file in tqdm(dataList):
    structure = torch.load(os.path.join(cfg.fileDir, file))
    structure = structure.to(device)
    pred = model.encoder(structure.x, structure.edge_index).to('cpu')
    pred = pred.detach().sum(dim=0) / len(pred)
    np.save(os.path.join(cfg.vectorDir, file), pred.numpy())
    # torch.save(pred.numpy(), os.path.join(cfg.vectorDir, file))
    # pred = pred.data
    # print(os.path.join(cfg.vectorDir, file))
    # print(pred)
    # print(pred.numpy())
    # print(torch.sum(pred, dim=0) / len(pred))
