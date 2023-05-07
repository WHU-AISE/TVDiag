from model.backbone.GAT import *
import torch
import torch.nn.functional as F
import torch.nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hiddens, out_dim, device='cpu'):
        super(Encoder, self).__init__()
        self.conv1 = dglnn.TAGConv(in_dim, hiddens, activation=torch.relu).to(device)
        self.conv2 = dglnn.TAGConv(hiddens, out_dim, activation=torch.relu).to(device)
        self.pool = dglnn.MaxPooling()

        self.out_dim = out_dim

    def forward(self, g, h, pool=False):
        x = self.conv1(g, h)
        x = self.conv2(g, x)
        if pool:
            x = self.pool(g, x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, num_heads=4, device='cpu'):
#         super(Encoder, self).__init__()
#         self.conv1 = dglnn.GATv2Conv(in_dim, hidden_dim, num_heads=num_heads, activation=F.relu).to(device)  # F.relu
#         self.conv2 = dglnn.GATv2Conv(hidden_dim * num_heads, out_dim, num_heads=num_heads, activation=F.relu).to(device)
#         # self.pool = dglnn.AvgPooling()
#         self.pool = dglnn.MaxPooling()
#         self.out_dim = out_dim
#
#     def forward(self, g, h, pool=False):
#         h = self.conv1(g, h)
#         hh = torch.zeros(h.shape[0], h.shape[1] * h.shape[2])
#         # 中间层多头特征拼接
#         for i in range(h.shape[0]):
#             hh[i] = torch.cat([t for t in h[i]])
#         h = self.conv2(g, hh)
#         # 最后一层多头特征取平均值
#         h = torch.mean(h, dim=1)
#         if pool:
#             h = self.pool(g, h)
#         return h
