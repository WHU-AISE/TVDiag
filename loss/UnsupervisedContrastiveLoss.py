import torch
from torch import nn
import torch.nn.functional as F


class UspConLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb_i, emb_j):
        data_len = len(emb_i)
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        # calculate the cos similarity between each embedding pair
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, data_len)
        sim_ji = torch.diag(similarity_matrix, data_len)
        # sim_ij = torch.diag(similarity_matrix)
        # sim_ji = torch.diag(similarity_matrix)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        # exp(sim(zi, zj)/t)
        nominator = torch.exp(positives / self.temperature)
        # exp(sim(zi, zk)/t)

        negatives_mask = (~torch.eye(data_len * 2, data_len * 2, dtype=bool)).float()
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * data_len)
        return loss