import torch.nn as nn
from torch.functional import F
import dgl.nn.pytorch as dglnn

class SAGEEncoder(nn.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dim, 
                 out_dim,
                 feat_drop=0.3,
                 aggregator_type='mean',
                 num_layers=2):
        super(SAGEEncoder, self).__init__()

        self.num_layers=num_layers
        self.sage_layers = nn.ModuleList()
        self.activation = F.relu

        if num_layers == 1:
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
        else:
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    feat_drop=feat_drop,
                    activation=self.activation,
                    aggregator_type=aggregator_type,
                    bias=False
                )
            )
            # hidden layers
            for l in range(0, num_layers-2):
                self.sage_layers.append(
                    dglnn.SAGEConv(
                        in_feats=hidden_dim,
                        out_feats=hidden_dim,
                        feat_drop=feat_drop,
                        aggregator_type=aggregator_type,
                        activation=self.activation,
                        bias=False
                    )
                )
            self.sage_layers.append(
                dglnn.SAGEConv(
                    in_feats=hidden_dim,
                    out_feats=out_dim,
                    feat_drop=feat_drop,
                    aggregator_type=aggregator_type,
                    activation=None,
                    bias=False
                )
            )
        
        self.pool=dglnn.MaxPooling()

    def forward(self, g, x):
        for l in range(self.num_layers-1):
            x = self.sage_layers[l](g, x)
        e = self.sage_layers[-1](g, x) # node-level
        f = self.pool(g, e) # graph-level
        return f, e
