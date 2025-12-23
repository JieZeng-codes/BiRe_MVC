from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize
from utils.utils import build_affinity_matrix,compute_correlation_matrix
from typing import Optional
from BT_filtering import *


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.PReLU(),
            nn.Linear(500, 500),
            nn.PReLU(),
            nn.Linear(500, 2000),
            nn.PReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.PReLU(),
            nn.Linear(2000, 500),
            nn.PReLU(),
            nn.Linear(500, 500),
            nn.PReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        xr = self.decoder(x)
        return xr


class ClusteringLayer(nn.Module):
    def __init__(self, class_num, hidden_dimension, alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):

        super(ClusteringLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.class_num = class_num
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.class_num, self.hidden_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch):
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)



class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, out_feature, class_num, neighbor_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.decs = []
        self.embed_projs = []
        self.clusters = []
        self.cluster_projs = []
        self.view = view
        self.class_num = class_num
        self.device = device
        self.neighbor_num = neighbor_num
        self.alpha = 1.0
        self.fusion_fea_adj = None
        self.ws = []

        for v in range(view):
            encoder = Encoder(input_size[v], feature_dim)
            decoder = Decoder(input_size[v], feature_dim)
            self.encoders.append(encoder.to(device))
            self.decoders.append(decoder.to(device))
            self.embed_projs.append(nn.Sequential(nn.Linear(feature_dim, out_feature)))


        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.embed_projs = nn.ModuleList(self.embed_projs)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=out_feature * view, nhead=1,dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.embed_proj = nn.Sequential(nn.Linear(feature_dim, out_feature))
        self.fusion_proj = nn.Sequential(nn.Linear(out_feature * view, out_feature))
        self.cluster_proj = nn.Sequential(
            nn.Linear(out_feature, out_feature),
            nn.PReLU(),
            nn.Linear(out_feature, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, xs):
        hs = []
        zs = []
        xrs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            f = self.embed_proj(z)
            hs.append(f)
            zs.append(z)
            xrs.append(xr)
        return xrs, hs, zs

    def forward_fusion(self, xs,args):
        hs = []
        weights = []
        adjs = []

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.embed_proj(z)
            adj = build_affinity_matrix(h, self.neighbor_num).to(self.device)
            adjs.append(adj)
            hs.append(h)



        corr_matrix = compute_correlation_matrix(hs)

        for v in range(self.view):
            corr_sum = torch.sum(corr_matrix[v, :]) + 1e-8
            weight = 1 / corr_sum
            weights.append(weight)
        weights = torch.nn.functional.softmax(torch.tensor(weights), dim=0)
        weighted_hs = [hs[v] * weights[v] for v in range(self.view)]
        normalized_hs = [normalize(h, dim=1) for h in weighted_hs]


        cat_feautre = torch.cat(normalized_hs, dim=1)
        cat_feautre = torch.unsqueeze(cat_feautre, dim=1)
        fusion_fea = self.TransformerEncoderLayer(cat_feautre)
        fusion_fea = torch.squeeze(fusion_fea, dim=1)
        norm_fusion_fea = normalize(self.fusion_proj(fusion_fea), dim=1)

        return hs, norm_fusion_fea, adjs

