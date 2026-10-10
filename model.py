from torch.nn import Sequential, Linear, ReLU
from dgl.nn import EdgeWeightNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from utils import *

EOS = 1e-10
norm = EdgeWeightNorm(norm='both')


class GCL(nn.Module):
    def __init__(self, nlayers, nlayers_proj, in_dim, emb_dim, proj_dim, dropout, sparse):
        super(GCL, self).__init__()

        self.encoder1 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)
        self.encoder2 = SGC(nlayers, in_dim, emb_dim, dropout, sparse)

        if nlayers_proj == 1:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim))
        elif nlayers_proj == 2:
            self.proj_head1 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))
            self.proj_head2 = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True), Linear(proj_dim, proj_dim))

        # Z = alpha Z^L + beta Z^H; softmax keeps both weights positive and normalized.
        self.coef_logits = nn.Parameter(torch.zeros(2))  # softmax -> [α, β]

        # Discriminator score D(z_i, g) = sigmoid(z_i W g^T).
        self.disc_W = nn.Parameter(torch.empty(proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.disc_W)

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")


    def get_embedding(self, x, a1, a2, source='all'):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        return torch.cat((emb1, emb2), dim=1)


    def get_embedding_comb(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        zL = self.proj_head1(emb1)
        zH = self.proj_head2(emb2)
        coef = torch.softmax(self.coef_logits, dim=0)
        return coef[0] * zL + coef[1] * zH




    def get_projection(self, x, a1, a2):
        emb1 = self.encoder1(x, a1)
        emb2 = self.encoder2(x, a2)
        proj1 = self.proj_head1(emb1)
        proj2 = self.proj_head2(emb2)
        return torch.cat((proj1, proj2), dim=1)

    def _combine(self, zL: torch.Tensor, zH: torch.Tensor):
        """Z = α Z^L + β Z^H; g = mean(Z)."""
        coef = torch.softmax(self.coef_logits, dim=0)  # [α, β]
        Z = coef[0] * zL + coef[1] * zH               # (N,d)
        g = Z.mean(dim=0)                              # (d,)
        return Z, g, coef

    def _disc_logits(self, z: torch.Tensor, g: torch.Tensor):
        """logits_i = z_i W g^T, shape (N,)"""
        wg = self.disc_W @ g           # (d,)
        return z @ wg                  # (N,)

    def contrastive_loss(self, zL, zH, zL_neg, zH_neg):
        """
        1/(4N) * sum_i [log D(z_i^L,g) + log(1-D(z~_i^L,g)) + log D(z_i^H,g) + log(1-D(z~_i^H,g))]
        我们用 BCEWithLogitsLoss 来最小化 -Eq.(4)
        """
        _, g, _ = self._combine(zL, zH)

        pos_L = self._disc_logits(zL, g)
        neg_L = self._disc_logits(zL_neg, g)
        pos_H = self._disc_logits(zH, g)
        neg_H = self._disc_logits(zH_neg, g)

        ones = torch.ones_like(pos_L)
        zeros = torch.zeros_like(neg_L)

        loss_L = self.bce(torch.cat([pos_L, neg_L], dim=0),
                          torch.cat([ones, zeros], dim=0))
        loss_H = self.bce(torch.cat([pos_H, neg_H], dim=0),
                          torch.cat([ones, zeros], dim=0))

        return (loss_L + loss_H) / 2.0


    def forward(self, x1, a1, x2, a2):
        emb1 = self.encoder1(x1, a1)
        emb2 = self.encoder2(x2, a2)
        proj1 = self.proj_head1(emb1)   # Z^L
        proj2 = self.proj_head2(emb2)   # Z^H

        # Corruption: shuffle node features to construct negative samples.
        perm = torch.randperm(x1.size(0), device=x1.device)
        x1_corrupt = x1[perm]
        x2_corrupt = x2[perm]  # 把行顺序打乱,把“第 i 个 spot 的特征”换成“另一个随机 spot 的特征”

        emb1_neg = self.encoder1(x1_corrupt, a1)
        emb2_neg = self.encoder2(x2_corrupt, a2)
        proj1_neg = self.proj_head1(emb1_neg)
        proj2_neg = self.proj_head2(emb2_neg)

        return self.contrastive_loss(proj1, proj2, proj1_neg, proj2_neg)


class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, input_dim, alpha, sparse, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha


    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h


    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2


    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.cuda()
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()


    # def weight_forward(self, features, edges):
    #     embeddings = self.get_node_embedding(features)
    #     edges_weights_raw = self.get_edge_weight(embeddings, edges)
    #     weights_lp = self.gumbel_sampling(edges_weights_raw)
    #     weights_hp = 1 - weights_lp
    #     return weights_lp, weights_hp

    def weight_forward(self, features, edges, force_uniform=False):
        if force_uniform:
            num_edges = edges.shape[1]
            weights_lp = torch.ones(num_edges, device=features.device)
            weights_hp = torch.zeros(num_edges, device=features.device)
            return weights_lp, weights_hp

        embeddings = self.get_node_embedding(features)
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp



    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes).cuda()
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes).cuda()
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            mask = torch.zeros(adj_lp.shape).cuda()
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes).cuda() - adj_hp * mask * self.alpha
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes).cuda())) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp

            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes, device='cuda')
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes).cuda())) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= - self.alpha
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp


    # def forward(self, features, edges):
    #     weights_lp, weights_hp = self.weight_forward(features, edges)
    #     adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
    #     return adj_lp, adj_hp, weights_lp, weights_hp

    def forward(self, features, edges, force_uniform=False):
        weights_lp, weights_hp = self.weight_forward(features, edges, force_uniform)
        adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
        return adj_lp, adj_hp, weights_lp, weights_hp



class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse

        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    def forward(self, x, g):
        x = torch.relu(self.linear(x))

        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x
