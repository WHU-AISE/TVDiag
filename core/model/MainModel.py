import torch
from torch import nn
import dgl
from config.exp_config import Config
from core.model.Classifier import Classifier
from core.model.Voter import Voter
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, config: Config):
        super(MainModel, self).__init__()

        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.encoders[modality] = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop
            )

        fti_fuse_dim = len(config.modalities) * config.graph_out
        rcl_fuse_dim = len(config.modalities) * config.graph_out

        self.locator = Voter(rcl_fuse_dim,
                             hiddens=config.linear_hidden,
                             out_dim=1)
        self.typeClassifier = Classifier(in_dim=fti_fuse_dim,
                                         hiddens=config.linear_hidden,
                                         out_dim=config.ft_num)

    def forward(self, batch_graphs):
        fs, es = {}, {}
        for modality, encoder in self.encoders.items():
            x_d = batch_graphs.ndata[modality]
            f_d, e_d = encoder(batch_graphs, x_d) # graph-level, node-level
            fs[modality] = f_d
            es[modality] = e_d


        f = torch.cat(tuple(fs.values()), dim=1)

        # failure type identification
        type_logit = self.typeClassifier(f)

        # root cause localization
        e = torch.cat(list(es.values()), dim=1)
        root_logit = self.locator(e)

        return fs, es, root_logit, type_logit


    def message_aggregator(self, batch_graphs):
        fs, es = {}, {}
        for modality, encoder in self.encoders.items():
            x_d = batch_graphs.ndata[modality]
            f_d, e_d = encoder(batch_graphs, x_d) # graph-level, node-level
            fs[modality] = f_d
            es[modality] = e_d

        f = torch.cat(tuple(fs.values()), dim=1)
        e = torch.cat(tuple(es.values()), dim=1)
        return f, e
