import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = GCNConv(11, 16)
        self.conv2 = GCNConv(16, 32)

        self.unconv1 = GCNConv(32, 16)
        self.unconv2 = GCNConv(16, 11)

        # self.encoder = nn.Sequential(
        #     GCNConv(8, 16),
        #     GCNConv(16, 32)
        # )

        # self.decoder = nn.Sequential(
        #     GCNConv(32, 16),
        #     GCNConv(16, 8)
        # )

    # def forward(self, x):
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = f.relu(self.conv1(x, edge_index))
        x = f.relu(self.conv2(x, edge_index))
        x = f.relu(self.unconv1(x, edge_index))
        x = f.relu(self.unconv2(x, edge_index))
        # x = self.encoder(x)
        # print(x.shape)
        # x = self.decoder(x)
        # print(x.shape)
        return x
