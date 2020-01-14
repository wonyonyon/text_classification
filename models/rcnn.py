# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import yaml


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

        self.lstm = nn.LSTM(config['embedding_size'], config['hidden_size'], num_layers=2, bidirectional=True)
        self.hidden_layer = nn.Linear(2 * config['hidden_size'] + config['embedding_size'], 2 * config['hidden_size'])
        self.fc = nn.Linear(2 * config['hidden_size'], config['num_classes'])
        self.max_pool_layer = nn.MaxPool1d(config['max_seq_len'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x_ = self.embedding(x)
        output, (h, c) = self.lstm(x_)
        output_plus = self.hidden_layer((torch.cat((output, x_), 2)))
        output_plus = F.relu(output_plus).permute(0, 2, 1)
        output_plus = self.max_pool_layer(output_plus).squeeze()

        return self.fc(output_plus)


