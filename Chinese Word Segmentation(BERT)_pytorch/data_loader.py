# -*- coding: utf-8 -*-
# author: Qinghe Li
# date: 2019/10/11

import os
import torch
import random
import numpy as np
from transformers.tokenization_bert import BertTokenizer

class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        labels = self.load_labels()
        self.label2idx = {label: idx for idx, label in enumerate(labels)}
        self.idx2label = {idx: label for idx, label in enumerate(labels)}
        params.label2idx = self.label2idx
        params.idx2label = self.idx2label
        self.label_pad_idx = self.label2idx['[PAD]']

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

    def load_labels(self):
        labels = []
        file_path = os.path.join(self.data_dir, 'labels.txt')
        with open(file_path, 'r') as file:
            for label in file:
                labels.append(label.strip())
        return labels

    def load_sentences_labels(self, sentences_file, labels_file, dict):
        sentences = []
        labels = []
        with open(sentences_file, 'r', encoding='gb18030') as fs:
            lines = fs.readlines()
            for line in lines:
                tokens = line.strip().split()
                tokens.insert(0, '[CLS]')
                tokens.append('[SEP]')
                tokens_to_id =  self.tokenizer.convert_tokens_to_ids(tokens)
                sentences.append(tokens_to_id)

        with open(labels_file, 'r', encoding='gb18030') as fl:
            lines = fl.readlines()
            for line in lines:
                label = line.strip().split()
                label.insert(0, '[CLS]')
                label.append('[SEP]')
                label_to_id = [self.label2idx[key] for key in label]
                labels.append(label_to_id)

        # checks to ensure there is a label for each token
        assert len(sentences) == len(labels)
        for i in range(len(sentences)):
            assert len(labels[i]) == len(sentences[i])

        # storing sentences and labels in dict d
        dict['data'] = sentences
        dict['labels'] = labels
        dict['size'] = len(sentences)

    def load_data(self, data_type):
        data = {}
        
        sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
        labels_path = os.path.join(self.data_dir, data_type, 'labels.txt')
        self.load_sentences_labels(sentences_file, labels_path, data)

        return data

    def data_iterator(self, data, shuffle=False):
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size'] // self.batch_size):
            # fetch sentences and labels
            sentences = [data['data'][idx] for idx in order[i*self.batch_size: (i+1)*self.batch_size]]
            labels = [data['labels'][idx] for idx in order[i*self.batch_size: (i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_labels = self.label_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][: cur_len] = sentences[j]
                    batch_labels[j][: cur_len] = labels[j]
                else:
                    batch_data[j] = sentences[j][: max_len]
                    batch_labels[j] = labels[j][: max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
    
            yield batch_data, batch_labels
