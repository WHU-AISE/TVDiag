
import torch.nn as nn 
from core.model.backbone.FC import FullyConnected


class Voter(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens=[64, 128], device='cpu'):
        super(Voter, self).__init__()

        self.net = FullyConnected(in_dim, out_dim, hiddens).to(device)

    def forward(self, h):
        return self.net(h)
