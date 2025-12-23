import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import combinations
from math import comb
from utils.utils import off_diagonal


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, gamma,lambda1, lambda2, lambda3, eta, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device
        self.temperature_f = temperature_f
        self.softmax = nn.Softmax(dim=1)
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.eta = eta
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def BarlowTwins_multi(self, device, hs, filter_dim):
        Batch_size = self.batch_size
        bn = nn.BatchNorm1d(filter_dim, affine=False).to(device)
        BT = torch.tensor([0]).to(device).float()
        num_view = len(hs)
        V = range(num_view)

        if num_view != 0:
            for combn in combinations(V, 2):
                c = bn(hs[combn[0]]).t().mm(bn(hs[combn[1]]))
                c.div_(Batch_size)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()

                BT += on_diag + self.gamma * off_diag
            BT = BT / comb(num_view, 2)

        return BT

    def feature_loss(self, zi, z, w, y_pse):
        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        N = z.size(0)
        w = w + torch.eye(N, dtype=int).to(w.device)
        positive_loss = (w & y_pse) * cross_view_distance
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f
        positive_loss = -torch.sum(positive_loss)
        negated_w = w ^ True
        negated_y = y_pse ^ True
        SMALL_NUM = torch.log(torch.tensor(1e-45)).to(zi.device)
        negtive_cross = (negated_w & negated_y) * cross_view_distance
        negtive_cross[negtive_cross == 0.] = SMALL_NUM
        negtive_inter = (negated_w & negated_y) * inter_view_distance
        negtive_inter[negtive_inter == 0.] = SMALL_NUM
        negtive_similarity = torch.cat((negtive_inter, negtive_cross), dim=1) / self.temperature_f
        negtive_loss = torch.logsumexp(negtive_similarity, dim=1, keepdim=False)
        negtive_loss = torch.sum(negtive_loss)
        return self.lambda2*(positive_loss + negtive_loss) / N


    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels,class_num):

        d_q = q_centers.mm(q_centers.T) / self.temperature_f
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_f
        d_q = d_q.float()
        d_q[torch.arange(class_num), torch.arange(class_num)] = d_k
        zero_classes = torch.arange(class_num).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),class_num), dim=0) == 0]
        mask = torch.zeros((class_num, class_num), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((class_num, class_num))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, class_num - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(class_num, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (class_num - len(zero_classes))

        return self.lambda1 *loss

    def forword_feature(self, h_i, h_j):
        feature_size, _ = h_i.shape
        N = 2 * feature_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, feature_size)
        sim_j_i = torch.diag(sim, -feature_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss_contrast = self.criterion(logits, labels)
        loss_contrast /= N

        return self.lambda3 * loss_contrast


