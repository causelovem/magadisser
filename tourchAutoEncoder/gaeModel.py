import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = GCNConv(11, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 50)

        self.unconv1 = GCNConv(50, 32)
        self.unconv2 = GCNConv(32, 16)
        self.unconv3 = GCNConv(16, 11)

    def encoder(self, x, edge_index):
        x = f.relu(self.conv1(x, edge_index))
        x = f.relu(self.conv2(x, edge_index))
        # x = f.relu(self.conv3(x, edge_index))
        x = self.conv3(x, edge_index)

        return x

    def decoder(self, x, edge_index):
        x = f.relu(self.unconv1(x, edge_index))
        x = f.relu(self.unconv2(x, edge_index))
        # x = f.relu(self.unconv3(x, edge_index))
        x = self.unconv3(x, edge_index)

        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)

        return x
