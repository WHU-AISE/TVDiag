import torch
from torch import nn

from model.Classifier import Classifyer
from model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args

        self.metric_encoder = Encoder(in_dim=args['metric_embedding_dim'],
                                      hiddens=args['graph_hidden'],                                
                                      out_dim=args['graph_out'])
        self.trace_encoder = Encoder(in_dim=args['trace_embedding_dim'],
                                     hiddens=args['graph_hidden'],
                                      out_dim=args['graph_out'])
        self.log_encoder = Encoder(in_dim=args['log_embedding_dim'],
                                   hiddens=args['graph_hidden'],
                                   out_dim=args['graph_out'])
        fuse_dim = self.metric_encoder.out_dim + self.trace_encoder.out_dim + self.log_encoder.out_dim

        self.locator = Classifyer(in_dim=fuse_dim, 
                                  hiddens=args['linear_hiddens'],
                                  out_dim=args['N_I'])
        self.typeClassifier = Classifyer(in_dim=fuse_dim,
                                         hiddens=args['linear_hiddens'],
                                         out_dim=args['N_A'])

    def forward(self, batch_graphs):
        metric_feats = batch_graphs.ndata['metrics']
        trace_feats = batch_graphs.ndata['traces']
        log_feats = batch_graphs.ndata['logs']

        metric_embs = self.metric_encoder(batch_graphs, metric_feats, pool=True)
        trace_embs = self.trace_encoder(batch_graphs, trace_feats, pool=True)
        log_embs = self.log_encoder(batch_graphs, log_feats, pool=True)

        embs = torch.cat((metric_embs, trace_embs, log_embs), dim=1)
        root_logit = self.locator(embs)
        type_logit = self.typeClassifier(embs)

        return (metric_embs, trace_embs, log_embs), root_logit, type_logit