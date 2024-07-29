
from torch import nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential
from core.model.backbone.gatv2 import GATEncoder
from core.model.backbone.SGC import SGCEncoder
from core.model.backbone.sage import SAGEEncoder
from core.model.backbone.cnn1d import CNN1dEncoder
import dgl.nn.pytorch as dglnn

class Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 graph_hidden_dim, 
                 out_dim,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 aggregator='mean'):
        super(Encoder, self).__init__()

        # word CNN
        # self.sequential_encoder = CNN1dEncoder(
        #     in_dim=1,
        #     hidden_dim=seq_hidden,
        #     kernel_size=3,
        #     dropout=0.2
        # ).to(device)

        # feature aggregation
        # self.graph_encoder = GATEncoder(
        #     in_dim=in_dim,
        #     out_dim=out_dim,
        #     hidden_dim=graph_hidden_dim,
        #     num_layers=2,
        #     heads=[8,1],
        #     feat_drop=feat_drop,
        #     attn_drop=attn_drop
        # )

        # self.graph_encoder = SGCEncoder(
        #     in_dim=in_dim,
        #     out_dim=out_dim,
        #     hidden_dim=graph_hidden_dim,
        #     num_layers=2,
        #     k=2
        # ).to(device)

        self.graph_encoder = SAGEEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=3,
            feat_drop=feat_drop,
            aggregator_type=aggregator
        )



        # self.graph_encoder = Sequential(
        #     dglnn.TAGConv(in_dim, graph_hidden_dim, activation=torch.relu).to(device),
        #     dglnn.TAGConv(graph_hidden_dim, out_dim, activation=torch.relu).to(device),
        #     dglnn.MaxPooling()
        # )
        # self.out_dim = out_dim

    def forward(self, g, x):
        # h = self.sequential_encoder(x)
        f = self.graph_encoder(g, x)
        return f