import torch
import torch.nn.functional as F
from torch import nn

# https://doi.org/10.1007/s13042-022-01622-7
# https://blog.csdn.net/wf19971210/article/details/116715880

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.device = device

    def forward(self, embeddings, labels):
        n = labels.shape[0]  # batch
        # similarity_matrix = torch.matmul(features, features.T)
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        mask = torch.ones_like(similarity_matrix) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask) - mask

        mask_dj = torch.ones(n, n) - torch.eye(n, n)
        mask_dj = mask_dj.to(self.device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_dj

        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim

        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend

        loss_partial = -torch.log(mask_no_sim + torch.div(sim, sim_sum) + torch.eye(n, n).to(self.device))

        nonzero_count = len(torch.nonzero(loss_partial))
        if nonzero_count == 0:
            nonzero_count = 1

        loss = torch.sum(torch.sum(loss_partial, dim=1)) / nonzero_count

        return loss



