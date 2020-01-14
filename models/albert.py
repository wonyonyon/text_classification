# -*- coding: utf-8 -*-
# -------------------------------------
# Description: CNN model for text classification
# @Author: tiniwu


import torch.nn as nn
import torch
from transformers import AlbertModel, AlbertForSequenceClassification


class TextModel(nn.Module):
    def __init__(self, config, weights=None):
        super(TextModel, self).__init__()
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        self.albert = AlbertModel.from_pretrained(config['albert']['bert_path'])

        for param in self.albert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config['albert']['hidden_size'], config['num_classes'])

    def forward(self, x):
        token_ids = x[0]
        masks = x[1]
        outputs = self.albert(token_ids, attention_mask=masks)
        return self.fc(outputs[1])
