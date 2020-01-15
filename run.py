# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu

from __future__ import absolute_import, division, print_function
import time
import torch
import yaml
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm, trange
import logging
import os
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import random
from importlib import import_module
from preprocessing.data_helper import TextProcessor, BertTextProcessor
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn import metrics
from datetime import timedelta
from transformers import AlbertTokenizer, BertTokenizer, AdamW

logger = logging.getLogger(__name__)


def set_seed(args):
    """
    set seed
    :return:
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(args, config, model, train_dataset, dev_dataset, is_bert=False):
    start_time = time.time()
    model.train()

    # generate dataset and data loader
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)
    writer = SummaryWriter(log_dir=config['log_path'] + '/' + args.model_name)

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      "weight_decay": 0.01},
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.01}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=float(config['learning_rate']), eps=float(config['adam_epsilon']))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
    total_batch = 0
    dev_best_loss = float('inf')
    lass_improved = 0
    flag = False

    for epoch in range(config['epochs']):
        print('Epoch [{}/{}]'.format(epoch + 1, config['epochs']))
        # epoch_iterator = tqdm(train_loader, desc="Iteration", disable=False)
        for i, batch in enumerate(train_loader):
            if is_bert:
                inputs, labels = (batch[0], batch[1]), batch[2]
            else:
                inputs, labels = batch[0], batch[1]
            # token_id = token_id.to(args.device)
            # label = token_id.to(args.device)
            print(inputs, labels)
            outputs = model(inputs)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(labels.data.cpu(), predict)
                if is_bert:
                    dev_acc, dev_loss = evaluate(config, model, dev_dataset, is_test=False, is_bert=True)
                else:
                    dev_acc, dev_loss = evaluate(config, model, dev_dataset, is_test=False)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    lass_improved = total_batch
                    torch.save(model.state_dict(), config['save_model_path'] + args.model_name + '.ckpt')
                    # logger.info("save step {} model to {}".format(total_batch, config['save_model_path']))

                cost_time = timedelta(seconds=int(round(time.time() - start_time)))
                logger_msg = 'Iter: {0:>6}\tTrain Loss: {1:>5.4}\tTrain Acc: {2:>6.2%}\tVal Loss: {3:>5.6}\tVal Acc: {4:>6.2%}\tCost Time: {5}'
                print(logger_msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, cost_time))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1

            if total_batch - lass_improved > 1000:
                logger.info("evaluate loss utils not optimizer, auto stop...")
                flag = True
                break
        if flag:
            break
    writer.close()


def evaluate(config,  model, dataset, is_test=False, is_bert=False):
    model.eval()
    total_loss = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    # generate dataset and data loader
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler)
    with torch.no_grad():
        for batch in data_loader:
            if is_bert:
                inputs, labels = (batch[0], batch[1]), batch[2]
            else:
                inputs, labels = batch[0], batch[1]
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)
            labels_all = np.append(labels_all, labels)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if is_test:
        class_list = [line.strip() for line in open(config['label_file'], 'r', encoding='utf-8').readlines()]
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, total_loss / len(data_loader), report, confusion

    return acc, total_loss/len(data_loader)


def test(args, config, model, dataset, is_bert=True):
    model.load_state_dict(torch.load(config['save_model_path'] + args.model_name + '.ckpt'))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, dataset, is_test=True, is_bert=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print("Time usage:", time.time() - start_time)


def main():
    parser = argparse.ArgumentParser(description='Chinese Text Classification By pytorch')
    parser.add_argument('--model_name', type=str, required=True, help='choose a model: cnn, rnn, transformer')
    # parser.add_argument("--model_conf", default=None, required=True, help="model config file.")
    parser.add_argument("--data_conf", default=None, required=True, help="data config file.")
    parser.add_argument("--max_len", default=32, type=int, help="The max sequence length.")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action='store_true', help="use cuda or not")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = torch.cuda.device_count()

    data_config = yaml.load(open(args.data_conf))
    if args.model_name in ['cnn', 'rnn', 'rcnn','rnn_attention', 'transformer']:
        processor = TextProcessor(data_config)

        text_cls_model = import_module('models.' + args.model_name)
        data_config['vocab_size'] = len(processor.word_to_id)
        model = text_cls_model.TextModel(data_config, weights=processor.word_embedding).to(args.device)
        set_seed(args)
        init_network(model)

        tokens_id, labels, _ = processor.get_train_examples(data_config['data_dir'])
        train_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(labels))

        tokens_id, labels, _ = processor.get_dev_examples(data_config['data_dir'])
        dev_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(labels))

        tokens_id, labels, _ = processor.get_test_examples(data_config['data_dir'])
        test_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(labels))
        train(args, data_config, model, train_dataset, dev_dataset)
        test(args, data_config, model, test_dataset)
    else:
        processor = BertTextProcessor(data_config, BertTokenizer)
        text_cls_model = import_module('models.' + args.model_name)
        model = text_cls_model.TextModel(data_config, weights=None).to(args.device)
        set_seed(args)
        init_network(model)

        tokens_id, masks, labels, _ = processor.get_train_examples(data_config['data_dir'])
        train_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(masks), torch.from_numpy(labels))

        tokens_id, masks, labels, _ = processor.get_dev_examples(data_config['data_dir'])
        dev_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(masks), torch.from_numpy(labels))

        tokens_id, masks, labels, _ = processor.get_test_examples(data_config['data_dir'])
        test_dataset = TensorDataset(torch.from_numpy(tokens_id), torch.from_numpy(masks), torch.from_numpy(labels))
        train(args, data_config, model, train_dataset, dev_dataset, is_bert=True)
        test(args, data_config, model, test_dataset, is_bert=True)


if __name__ == '__main__':
    main()

# f = open('../conf/data.yaml')
# config = yaml.load(f)
# processor = TextProcessor(config)
# x, y, _ = processor.get_train_examples(config['data_dir'])
# train_data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
# train_loader = DataLoader(train_data, shuffle=True, batch_size=1)
#
# epochs = 10
# # for epoch in epochs:
# for i, (x, y) in enumerate(train_loader):
#     if i < 5:
#         print(x, y)
