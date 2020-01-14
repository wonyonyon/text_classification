# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu

import re
from collections import Counter
import numpy as np
import jieba
import pickle
import os


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(train_path, vocab_path, max_vocab_size=100000, min_freq=3):
    UNK, PAD = ['<UNK>'], ['<PAD>']
    docs = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.split('\t')) != 2: continue
            docs.extend(jieba.lcut(line.split('\t')[0]))

    vocab = UNK + PAD + [str(t[0]) for t in Counter(docs).most_common(max_vocab_size)[1:] if t[1] > min_freq]
    word2id = {word: idx for idx, word in enumerate(vocab)}
    pickle.dump(word2id, open(vocab_path, 'wb'))
    return word2id


def get_word_embedding(vec_path, save_path, word_2_id, embedding_size=300):
    embeddings = np.random.rand(len(word_2_id), int(embedding_size))
    with open(vec_path, 'r', encoding='utf-8') as f:
        for line in f:
            sps = line.split()
            if len(sps) != 301: continue
            if sps[0] in word_2_id:
                idx = word_2_id[sps[0]]
                embeddings[idx] = np.array([float(x) for x in sps[1:]], dtype='float32')
    f.close()
    np.save(save_path, embeddings)


if __name__ == '__main__':
    vocab_path = '../data/ThUCNews/words.pkl'
    # if os.path.exists(vocab_path):
    #     print('Thr vocab has already built.')
    # else:
    word2id = build_vocab('../data/ThUCNews/train.txt', vocab_path)
    # print(word2id.get('<PAD>'))
    get_word_embedding('../data/ThUCNews/sgns.sogou.word', '../data/ThUCNews/word2vec.npy', word_2_id=word2id)


