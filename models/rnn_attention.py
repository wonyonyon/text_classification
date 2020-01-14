# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
import torch.nn.functional as F
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

        self.lstm = nn.LSTM(config['embedding_size'], config['hidden_size'], num_layers=config['rnn']['num_layers'],
                            batch_first=True, bidirectional=True, dropout=config['dropout'])

        self.w = nn.Parameter(torch.Tensor(config['hidden_size'] * 2))
        self.fc = nn.Linear(2 * config['hidden_size'], config['num_classes'])
        self.dropout = nn.Dropout(config['dropout'])

    def attention_old(self, lstm_output, final_state):
        # hidden = final_state.view(-1, 2 * config['hidden_size'], 1)
        all_seq_state = torch.cat([state for state in final_state], 1).squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        attention_weights = torch.bmm(lstm_output, all_seq_state).squeeze(2)    # attention_w : [batch_size, n_step]
        soft_attention_w = F.softmax(attention_weights, 1).unsqueeze(2)
        attention_output = torch.bmm(lstm_output.transpose(1, 2), soft_attention_w).squeeze(2)

        return attention_output

    def attention(self, output):
        attention_weights = F.softmax(torch.matmul(output, self.w), dim=1).unsqueeze(-1)
        attention_output = torch.sum(attention_weights * output, 1)

        return attention_output

    def forward(self, x):
        x_ = self.dropout(self.embedding(x))
        output, (h, c) = self.lstm(x_)
        output = self.attention(output)
        return self.fc(output)

# test
# f = open('../conf/data.yaml')
# config = yaml.load(f)
# model = TextModel(config)
# print(model)