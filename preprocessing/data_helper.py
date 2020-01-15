# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu

import os
import pickle as pkl

import jieba
import numpy as np
from tqdm import tqdm
import torch
import logging

logger = logging.getLogger(__name__)
UNK, PAD, CLS = '[UNK]', '[PAD]', '[CLS]'


class BasicTextProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self, dataset_name):
        raise NotImplementedError()


class TextProcessor(BasicTextProcessor):
    def __init__(self, config):
        self.config = config
        if os.path.exists(config['vocab_file']):
            self.word_to_id = pkl.load(open(config['vocab_file'], 'rb'))
            self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        else:
            print('Please run utils.py to generate vocab file and word pre-trained embedding file.')
        self.word_embedding = self._get_word_embedding()

    def get_train_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "test.txt"))

    def get_labels(self, dataset_name):
        return open(dataset_name, 'r', encoding='utf-8').readlines()

    def _get_word_embedding(self):
        word_vec = torch.tensor(np.load(self.config['word_vec_file']).astype('float32'))
        logger.info('Load pre-train embedding file.')
        return word_vec

    def token_to_id(self, word):
        return self.word_to_id.get(word, self.word_to_id.get(UNK))

    def _read_file(self, input_file):
        data = []
        tokens_id = []
        labels = []
        seq_lens = []
        with open(input_file, 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                splits = line.strip('\n').split('\t')
                if len(splits) != 2:
                    continue
                else:
                    tokens = jieba.lcut(splits[0])
                    seq_len = len(tokens)
                    if seq_len > self.config['max_seq_len']:
                        tokens = tokens[:self.config['max_seq_len']]
                    else:
                        tokens.extend((self.config['max_seq_len'] - seq_len) * [PAD])
                        seq_len = self.config['max_seq_len']
                    seq_lens.append(seq_len)
                    labels.append(int(splits[-1]))
                    tokens_id.append([self.token_to_id(token) for token in tokens])
                    # data.append(([self.token_to_id(token) for token in tokens], int(splits[-1]), seq_len))
            return np.array(tokens_id), np.array(labels), seq_lens


class BertTextProcessor(BasicTextProcessor):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer.from_pretrained(self.config['bert_path'])

    def get_train_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self._read_file(os.path.join(data_dir, "test.txt"))

    def get_labels(self, dataset_name):
        return open(dataset_name, 'r', encoding='utf-8').readlines()

    def _read_file(self, input_file):
        tokens_id = []
        labels = []
        seq_lens = []
        masks = []
        with open(input_file, 'r', encoding="utf-8") as f:
            for line in tqdm(f):
                splits = line.strip('\n').split('\t')
                if len(splits) != 2:
                    continue
                else:
                    tokens = [CLS] + self.tokenizer.tokenize(splits[0])
                    seq_len = len(tokens)
                    if seq_len > self.config['max_char_len']:
                        tokens = tokens[:self.config['max_char_len']]
                        mask = [1] * self.config['max_char_len']
                    else:
                        tokens.extend((self.config['max_char_len'] - seq_len) * [PAD])
                        mask = [1] * seq_len + (self.config['max_char_len'] - seq_len) * [0]
                        seq_len = self.config['max_char_len']
                    seq_lens.append(seq_len)
                    labels.append(int(splits[-1]))
                    tokens_id.append([self.tokenizer.convert_tokens_to_ids(token) for token in tokens])
                    masks.append(mask)
                    # data.append(([self.token_to_id(token) for token in tokens], int(splits[-1]), seq_len))
            return np.array(tokens_id), np.array(masks), np.array(labels), seq_lens
