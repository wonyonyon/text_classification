# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
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

        self.lstm = nn.LSTM(config['embedding_size'], config['hidden_size'], num_layers=2, batch_first=True,
                            bidirectional=True, dropout=config['dropout'])

        self.output = nn.Linear(2 * config['hidden_size'], config['num_classes'])

    def forward(self, x):
        embedding_x = self.embedding(x)
        output, (h, c) = self.lstm(embedding_x)
        return self.output(output[:, -1, :])

# test
# f = open('../conf/model.yaml')
# config = yaml.load(f)
# model = TextRNN(config)
# print(model)
