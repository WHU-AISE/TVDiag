import torch
import torch.nn as nn
from torch.functional import F
import dgl.nn.pytorch as dglnn

class CNN1dEncoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dim,
                 kernel_size=3, 
                 dropout=0):
        super(CNN1dEncoder, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pool=dglnn.nn.AdaptiveMaxPool1d(1)

    def forward(self,  x): #[batch_size, T]
        x = torch.unsqueeze(x, 2) #[batch_size, T, 1]
        x = x.permute(0, 2, 1) #[batch_size, 1, T]
        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]
        out = self.pool(out)
        return torch.squeeze(out, 2)