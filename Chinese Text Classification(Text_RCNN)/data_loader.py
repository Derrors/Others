# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/06

import re
import os
import json
import torch
import numpy as np 
import random

from tqdm import trange
from gensim.models import Word2Vec


class Dataloader():
    '''
    数据加载器
    '''
    def __init__(self, word2vec_path, batch_size, embeddings_size, seq_length, device):
        self.word2vec = Word2Vec.load(word2vec_path)        # 预训练的词向量
        self.batch_size = batch_size
        self.embeddings_size = embeddings_size              # 词向量的维度
        self.seq_length = seq_length                        # 每个样本的句子长度
        self.device = device
        self.dict_labels = {}


    def load_data(self, data_file, shuffle=False, mode='train'):
        '''
        加载文本数据
        '''
        texts = []
        labels = []
        dict_labels = {}
        
        with open(data_file, 'r') as fj:
            data = json.load(fj, strict=False)
            for index, key in enumerate(data):
                t = trange(len(data[key]))
                for i in t:
                    doc = data[key][i]
                    sentences = []
                    for line in doc:
                        sentence = line.strip()
                        sentences.append(sentence)
                    texts.append(sentences)
                    labels.append(index)
                dict_labels[index] = key

        if mode == 'train':
            self.dict_labels = dict_labels

        if shuffle:                         # 数据乱序
            length = len(texts)
            order = list(range(length))
            random.seed(2019)
            random.shuffle(order)
            shuffle_texts  = [texts[idx] for idx in order]
            shuffle_labels = [labels[idx] for idx in order]
            return shuffle_texts, shuffle_labels
        else:
            return texts, labels
    
    def data_iterator(self, texts, labels, mode='train'):
        '''
        数据迭代器
        '''
        embeddings = []

        for doc in texts:
            doc_embedding = []
            doc_array = np.zeros((self.seq_length, self.embeddings_size), dtype=float)

            if len(doc) < self.seq_length:
                for sentence in doc:
                    sentence_embedding = 0.0
                    tokens = sentence.strip().split()
                    for token in tokens:
                        sentence_embedding += self.word2vec.wv[token]       # 将文本转换为词向量
                    doc_embedding.append(sentence_embedding)
                doc_array[: len(doc)] = doc_embedding[:]
            else:
                for i in range(self.seq_length):
                    sentence_embedding = 0.0
                    sentence = doc[i]
                    tokens = sentence.strip().split()
                    for token in tokens:
                        sentence_embedding += self.word2vec.wv[token]
                    doc_embedding.append(sentence_embedding)
                doc_array[:] = doc_embedding[:]
            embeddings.append(doc_array)

        # 若为验证集，则直接返回对应的向量表示
        if mode == 'valid':
            batch_embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
            batch_labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            return batch_embeddings, batch_labels
            
        # 若为训练集和测试集，则根据 batch_size 返回数据迭代器
        else:
            length = len(embeddings)
            for i in range(length // self.batch_size):
                batch_embeddings = embeddings[i * self.batch_size: (i + 1) * self.batch_size]
                batch_labels = labels[i * self.batch_size: (i + 1) * self.batch_size]

                batch_embeddings = torch.tensor(batch_embeddings, dtype=torch.float).to(self.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                yield batch_embeddings, batch_labels
