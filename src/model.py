# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""


import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from InOutGGNN import InOutGGNN
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv


class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, node_embedding, item_embedding_table, batch, num_count):
        sections = torch.bincount(batch)
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(node_embedding)))    # |V|_i * 1
        s_g_whole = num_count.view(-1, 1) * alpha * node_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        # print('s_h.shape', s_h.shape)
        
        # Eq(8)
        z_i_hat = torch.mm(s_h, item_embedding_table.weight.transpose(1, 0))
        
        return z_i_hat

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 100
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out

class Embedding2ScoreSAN(nn.Module):
    def __init__(self, hidden_size, san_blocks=3):
        super(Embedding2ScoreSAN, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.rn = Residual()
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.san_blocks = san_blocks

    def forward(self, node_embedding, item_embedding_table, batch, num_count):
        sections = torch.bincount(batch)
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i)    # repeat |V|_i times for the last node embedding

        # # Eq(6)
        # alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(node_embedding)))    # |V|_i * 1
        # s_g_whole = num_count.view(-1, 1) * alpha * node_embedding    # |V|_i * hidden_size
        # s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        # s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        # # print('origin s_g[0].shape', s_g[0].shape)

        s_g = []
        for node_embs in v_i:
            attn_output = node_embs.unsqueeze(0)
            for k in range(self.san_blocks):
                attn_output, attn_output_weights = self.multihead_attn(attn_output, attn_output, attn_output)
                attn_output = self.rn(attn_output)
            s_g.append(attn_output[0: 1, -1])
        # print('s_g[0].shape', s_g[0].shape)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        z_i_hat = torch.mm(s_h, item_embedding_table.weight.transpose(1, 0))

        
        
        return z_i_hat


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, use_san=False, use_gat=False):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.use_san = use_san
        self.use_gat = use_gat
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        if not use_gat:
            self.gated = InOutGGNN(self.hidden_size, num_layers=1)
        else:
            self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=4, negative_slope=0.2)
            self.gat2 = GATConv(4 * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        if not use_san:
            self.e2s = Embedding2Score(self.hidden_size)
        else:
            self.e2s = Embedding2ScoreSAN(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, data):
        x, edge_index, batch, edge_count, in_degree_inv, out_degree_inv, sequence, num_count = \
            data.x - 1, data.edge_index, data.batch, data.edge_count, data.in_degree_inv, data.out_degree_inv,\
            data.sequence, data.num_count
        # print('x.shape=', x.shape)
        # print('edge_index.shape=', edge_index.shape)
        # print('batch.shape=', batch.shape)
        # print('num_count.shape=', num_count.shape)
        embedding = self.embedding(x).squeeze()
        if not self.use_gat:
            hidden = self.gated(embedding, edge_index, [edge_count * in_degree_inv, edge_count * out_degree_inv])
        else:
            hidden = F.relu(self.gat1(embedding, edge_index))
            hidden = self.gat2(hidden, edge_index)

        return self.e2s(hidden, self.embedding, batch, num_count)
