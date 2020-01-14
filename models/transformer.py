# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import copy


class TextModel(nn.Module):
    def __init__(self, config, weights=None):
        super(TextModel, self).__init__()
        if weights is not None:
            # load pre-train embedding by three different ways
            # self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
            # self.embedding.weight = nn.Parameter(torch.FloatTensor(weights))
            # self.embedding.weight.requires_grad = False

            # pretrain_embedding = np.array(np.load(config['embedding_file']),dtype=float)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_embedding))

            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)

        else:
            self.embedding = nn.Embedding(config['vocab_size'], config['embedding_size'])

        self.position_embedding = PositionEncoding(config['embedding_size'], config['max_seq_len'])
        self.encoder = Encoder(config['transformer']['d_model'], config['transformer']['num_head'],
                               config['transformer']['hidden_size'], config['dropout'])

        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config['transformer']['num_encoder'])])

        self.fc1 = nn.Linear(config['max_seq_len'] * config['transformer']['d_model'], config['num_classes'])

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(dim_model, num_head, dropout)
        self.pos_ffn = PositionWiseFeedForward(dim_model, hidden_size, dropout)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.pos_ffn(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v):
        scale = k.size(-1) ** -0.5
        attention = torch.matmul(q, k.transpose(2, 3)) * scale
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.head_size = dim_model // self.num_head
        assert dim_model % self.num_head == 0

        self.q = nn.Linear(dim_model, num_head * self.head_size)
        self.k = nn.Linear(dim_model, num_head * self.head_size)
        self.v = nn.Linear(dim_model, num_head * self.head_size)
        self.fc = nn.Linear(num_head * self.head_size, dim_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self, x):
        batch_size = x.size(0)
        residual = x
        # x = self.layer_norm(x)

        Q = self.q(x).view(batch_size, -1, self.num_head, self.head_size)
        K = self.k(x).view(batch_size, -1, self.num_head, self.head_size)
        V = self.v(x).view(batch_size, -1, self.num_head, self.head_size)

        output = self.attention(Q, K, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.head_size)

        x = self.dropout(self.fc(output))

        x += residual
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden_size, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dim_model)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        residual = x
        # x = self.layer_norm(x)

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super(PositionEncoding, self).__init__()

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / hidden_size) for hid_j in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(max_seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        sinusoid_table = torch.FloatTensor(sinusoid_table).unsqueeze(0)
        self.register_buffer('pos_table', sinusoid_table)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

# test
# import yaml
# f = open('../conf/data.yaml')
# config = yaml.load(f)
# model = TextModel(config)
# print(model)
