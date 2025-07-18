import torch
import torch.nn as nn
import torch_geometric.utils
from torch_geometric.nn import GATConv
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F, LazyLinear
from torch import Tensor
import math

from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_sparse import to_torch_sparse


# full connected layer
def full_block(in_features, out_features, p_drop=0.0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )

class GATEncoder(nn.Module):

    def __init__(self, num_genes, latent_dim, num_heads=20
                 , dropout=0.4, fc=None):
        super(GATEncoder, self).__init__()
        # initialize parameter
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.gat_layer_1 = GATConv(
            in_channels=num_genes, out_channels=128,
            heads=num_heads,
            dropout=dropout,
            concat=True)
        in_dim2 = 128 * num_heads

        self.gat_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=latent_dim,
            heads=num_heads,
            concat=False)

        self.fc = fc

    def forward(self, x, edge_index):
        hidden_out1 = self.gat_layer_1(x, edge_index)
        hidden_out1 = F.relu(hidden_out1)
        hidden_out1 = F.dropout(hidden_out1, p=0.4, training=self.training)
        hidden_out2 = self.gat_layer_2(hidden_out1, edge_index)
        hidden_out2 = F.relu(hidden_out2)
        hidden_out2 = F.dropout(hidden_out2, p=0.4, training=self.training)
        embedding = hidden_out2
        # add project head
        if self.fc is not None:
            embedding = self.fc(embedding)
        return embedding


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, use_graph_transformer=False, num_genes=3000, num_node=90, dim=256, r=512, m=0.99, T=0.2, head=20,
                 mlp=False, n_layers=2, hidden_mlp_dims=64, hidden_dims=64, merge_weight=1, hidden_n_head=4):
        """
        dim: feature dimension
        r: queue size
        m: momentum for updating key encoder
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T
        self.merge_weight = merge_weight

        # create the encoders
        self.encoder_q = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)
        self.encoder_k = base_encoder(num_genes=num_genes, latent_dim=dim, num_heads=head)

        # if use graph transformer
        if use_graph_transformer:
            self.encoder_graph_q = GraphTransformer(n_layers=n_layers, input_dims=num_genes, num_nodes=num_node, hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims, output_dims=dim, hidden_n_head=hidden_n_head)
            self.encoder_graph_k = GraphTransformer(n_layers=n_layers, input_dims=num_genes, num_nodes=num_node, hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims, output_dims=dim, hidden_n_head=hidden_n_head)
        self.use_graph_transformer = use_graph_transformer

        # 1、create mlp
        if mlp:
            self.encoder_q.fc = nn.Sequential(full_block(dim, 512, 0.4), full_block(512, dim))
            self.encoder_k.fc = nn.Sequential(full_block(dim, 512, 0.4), full_block(512, dim))
        # initialize
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self.use_graph_transformer:
            for param_graph_q, param_graph_k in zip(self.encoder_graph_q.parameters(), self.encoder_graph_k.parameters()):
                param_graph_k.data = param_graph_k.data * self.m + param_graph_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k=None, edge_index=None, is_eval=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return embeddings (used for clustering)

        Output:
            logits, targets
        """

        if is_eval:
            k = self.encoder_k(im_q, edge_index)
            k = nn.functional.normalize(k, dim=1)

            if self.use_graph_transformer:
                k_graph = self.encoder_graph_k(im_q, edge_index)
                k_graph = nn.functional.normalize(k_graph, dim=1)
                k = k + k_graph * self.merge_weight
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k, edge_index)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            if self.use_graph_transformer:
                k_graph = self.encoder_graph_k(im_k, edge_index)
                k_graph = nn.functional.normalize(k_graph, dim=1)
                k = k_graph
                #k = k + k_graph * self.merge_weight

        # compute query features
        q = self.encoder_q(im_q, edge_index)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        if self.use_graph_transformer:
            q_graph = self.encoder_graph_q(im_q, edge_index)
            q_graph = nn.functional.normalize(q_graph, dim=1)
            q = q_graph
            #q = q + q_graph * self.merge_weight

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # N表示一个batch样本数
        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class GraphTransformer(nn.Module):
    def __init__(self, n_layers: int, input_dims: int, num_nodes: int, hidden_mlp_dims: int, hidden_dims: int,
                 output_dims: int, hidden_n_head: int, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim = output_dims
        self.batch_size = num_nodes
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims, hidden_mlp_dims), act_fn_in,
                                      nn.Linear(hidden_mlp_dims, hidden_dims), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(num_nodes, hidden_mlp_dims), act_fn_in,
                                      nn.Linear(hidden_mlp_dims, num_nodes), act_fn_in)

        self.tf_layers = nn.ModuleList([XETransformerLayer(dx=hidden_dims,
                                                            de=num_nodes,
                                                            n_head=hidden_n_head,
                                                            dim_ffX=hidden_dims,
                                                            dim_ffE=hidden_dims)
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims, hidden_mlp_dims), act_fn_out,
                                       nn.Linear(hidden_mlp_dims, output_dims))

        self.mlp_out_E = nn.Sequential(nn.Linear(num_nodes, hidden_mlp_dims), act_fn_out,
                                       nn.Linear(hidden_mlp_dims, num_nodes))

    def forward(self, im, edge_index):
        max_num_nodes = im.size(0)
        X, node_mask = to_dense_batch(x=im, max_num_nodes=max_num_nodes)
        new_dge_index, new_edge_attr = torch_geometric.utils.remove_self_loops(edge_index=edge_index)
        E = to_dense_adj(edge_index=new_dge_index, edge_attr=new_edge_attr, max_num_nodes=max_num_nodes)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        E = new_E

        X = self.mlp_in_X(X)

        for layer in self.tf_layers:
            X, E = layer(X, E, node_mask)

        X = self.mlp_out_X(X)
        X = X.squeeze(0)

        return X


class XETransformerLayer(nn.Module):
    def __init__(self, dx: int, de: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, node_mask: Tensor):
        newX, newE = self.self_attn(X, E, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        return X, E


class NodeEdgeBlock(nn.Module):
    def __init__(self, dx, de, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.e_final = LazyLinear(de)

    def forward(self, X, E, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Output E
        newE = Y.flatten(start_dim=3)
        newE = self.e_out(newE) * e_mask1 * e_mask2
        newE = newE.flatten(start_dim=2)
        newE = self.e_final(newE)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx
        newX = weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask

        return newX, newE


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
