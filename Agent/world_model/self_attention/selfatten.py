import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import os
import math


def masked_softmax(X, valid_len):
    """
    masked softmax for attention scores
    args:
        X: 3-D tensor, valid_len: 1-D or 2-D tensor
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(
                valid_len, repeats=shape[1], dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = X.reshape(-1, shape[-1])
        for count, row in enumerate(X):
            row[int(valid_len[count]):] = -1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        
        hidden_unit = 8
        # self.q_lin = nn.Sequential(
        #     nn.Linear(in_channels, hidden_unit),
        #     nn.LayerNorm(hidden_unit),
        #     nn.ReLU(),
        #     nn.Linear(hidden_unit, global_graph_width)
        # )
        # self.k_lin = nn.Sequential(
        #     nn.Linear(in_channels, hidden_unit),
        #     nn.LayerNorm(hidden_unit),
        #     nn.ReLU(),
        #     nn.Linear(hidden_unit, global_graph_width)
        # )
        # self.v_lin = nn.Sequential(
        #     nn.Linear(in_channels, hidden_unit),
        #     nn.LayerNorm(hidden_unit),
        #     nn.ReLU(),
        #     nn.Linear(hidden_unit, global_graph_width)
        # )
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)

        self._norm_fact = 1 / math.sqrt(in_channels)

    def forward(self, x):

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = nn.functional.softmax(scores, dim=-1)
        # scores = nn.Softmax(dim=-1)(torch.matmul(query,key.transpose(0, 1)) * self._norm_fact) 
        # print("x",x)
        # print("q",query)
        # print("k",key)
        # print("v",value)
        # print("key.transpose(0, 1)",key.transpose(0, 1))
        # print("debug",torch.matmul(query,key.transpose(0, 1)) * self._norm_fact)
        # print("scores",scores)
        # print("output",torch.matmul(scores,value))
        
        return torch.bmm(scores,value)
