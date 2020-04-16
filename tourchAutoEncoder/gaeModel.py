import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, BatchNorm


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.conv1 = GCNConv(88, 70)
        self.bn1 = BatchNorm(70)
        self.conv2 = GCNConv(70, 60)
        self.bn2 = BatchNorm(60)
        self.conv3 = GCNConv(60, 50)

        self.unconv1 = GCNConv(50, 60)
        self.bn3 = BatchNorm(60)
        self.unconv2 = GCNConv(60, 70)
        self.bn4 = BatchNorm(70)
        self.unconv3 = GCNConv(70, 88)

        # self.conv1 = GCNConv(88, 70)
        # self.conv2 = GCNConv(70, 60)
        # self.conv3 = GCNConv(60, 50)
        # self.conv4 = GCNConv(50, 40)

        # self.unconv1 = GCNConv(40, 50)
        # self.unconv2 = GCNConv(50, 60)
        # self.unconv3 = GCNConv(60, 70)
        # self.unconv4 = GCNConv(70, 88)

    def encoder(self, x, edge_index):
        x = f.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = f.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = f.relu(self.conv3(x, edge_index))
        x = self.conv3(x, edge_index)

        return x

    def decoder(self, x, edge_index):
        x = f.relu(self.unconv1(x, edge_index))
        x = self.bn3(x)
        x = f.relu(self.unconv2(x, edge_index))
        x = self.bn4(x)
        # x = f.relu(self.unconv3(x, edge_index))
        x = self.unconv3(x, edge_index)

        return x

    # def encoder(self, x, edge_index):
    #     x = f.relu(self.conv1(x, edge_index))
    #     x = f.relu(self.conv2(x, edge_index))
    #     x = f.relu(self.conv3(x, edge_index))
    #     x = self.conv4(x, edge_index)

    #     return x

    # def decoder(self, x, edge_index):
    #     x = f.relu(self.unconv1(x, edge_index))
    #     x = f.relu(self.unconv2(x, edge_index))
    #     x = f.relu(self.unconv3(x, edge_index))
    #     x = self.unconv4(x, edge_index)

    #     return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)

        return x
