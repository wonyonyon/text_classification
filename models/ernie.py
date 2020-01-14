# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
import numpy as np
import yaml


import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import yaml


class TextModel(nn.Module):
    def __init__(self, config, weights=None):
        super(TextModel, self).__init__()
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        self.bert = BertModel.from_pretrained(config['erine']['bert_path'])

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config['erine']['hidden_size'], config['num_classes'])

    def forward(self, x):
        token_ids = x[0]
        masks = x[1]
        _, pooled = self.bert(token_ids, attention_mask=masks)
        out = self.fc(pooled)
        return out