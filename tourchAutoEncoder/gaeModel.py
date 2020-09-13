# модель нейронной сети
# можно эксперементировать с разными слоями и активационными функциями

import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, BatchNorm


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = GCNConv(37, 30)
        # self.bn1 = BatchNorm(30)
        self.conv2 = GCNConv(30, 25)

        self.unconv1 = GCNConv(25, 30)
        # self.bn2 = BatchNorm(30)
        self.unconv2 = GCNConv(30, 37)

    # после обучения будет использоваться только энкодер
    def encoder(self, x, edge_index):
        x = f.relu(self.conv1(x, edge_index))
        # x = self.bn1(x)
        x = self.conv2(x, edge_index)

        return x

    def decoder(self, x, edge_index):
        x = f.relu(self.unconv1(x, edge_index))
        # x = self.bn2(x)
        x = self.unconv2(x, edge_index)

        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)

        return x


# import torch.nn as nn
# import torch.nn.functional as f
# from torch_geometric.nn import GCNConv, BatchNorm


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()

#         self.conv1 = GCNConv(33, 27)
#         # self.bn1 = BatchNorm(27)
#         self.conv2 = GCNConv(27, 20)

#         self.unconv1 = GCNConv(20, 27)
#         # self.bn2 = BatchNorm(27)
#         self.unconv2 = GCNConv(27, 33)

#     def encoder(self, x, edge_index):
#         x = f.relu(self.conv1(x, edge_index))
#         # x = self.bn1(x)
#         x = self.conv2(x, edge_index)

#         return x

#     def decoder(self, x, edge_index):
#         x = f.relu(self.unconv1(x, edge_index))
#         # x = self.bn2(x)
#         x = self.unconv2(x, edge_index)

#         return x

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.encoder(x, edge_index)
#         x = self.decoder(x, edge_index)

#         return x
