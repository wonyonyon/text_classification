# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch.nn.functional as F
import torch


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
        filter_sizes = config['cnn']['filter_sizes'].split(',')
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config['cnn']['num_filters'],
                       kernel_size=(int(filter_size), config['embedding_size'])) for filter_size in filter_sizes])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(len(filter_sizes) * config['cnn']['num_filters'], config['num_classes'])

    def forward(self, x):
        x = self.embedding(x)  # batch_size * seq_len * embedding_size (B*L*H)
        x = x.unsqueeze(1)  # [B*C*L*H]
        h = [conv(x) for conv in self.convs]  # [B*C*L*1] * K
        h = [F.relu(k).squeeze(3) for k in h]  # [B*C*L] * K
        h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h]  # [B * C] * K
        h = torch.cat(h, 1)  # B * [C * K]
        h = self.dropout(h)
        out = self.fc(h)

        return out



