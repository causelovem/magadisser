import numpy as np
import torch
import random
import os
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(23)


class readData(torch.utils.data.Dataset):
    def __init__(self, fileDir):
        self.fileDir = fileDir
        self.files = os.listdir(fileDir)
        self.threshold = 7

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            array = strucio.load_structure(os.path.join(self.fileDir, self.files[index]))
            cell_list = struc.CellList(array, cell_size=self.threshold)
            adjacency_matrix = cell_list.create_adjacency_matrix(self.threshold).astype(int)

            return adjacency_matrix
