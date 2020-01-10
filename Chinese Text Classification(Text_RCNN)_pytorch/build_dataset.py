# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/06

import os
import re
import json
from tqdm import trange
from fenci import Segment_Words
from gensim.models import word2vec

#配置原始文本数据路径
original_train_path = './original_data/train/'
original_test_path  = './original_data/test/'

corpus_path = './data/corpus.txt'
save_path = './data/word2vec.bin'

corpus = []


def str_filter(string):
    '''
    去除文本中的非中文字符
    '''
    chars = re.compile(r'[^\u4e00-\u9fa5]')
    return re.sub(chars, '', string)


def build_train_dataset(fenci):
    '''
    构建训练数据集
    '''
    print('Start to build training dataset...')
    dict_train = {}

    for root, _, files in os.walk(original_train_path): 
        if files != []:
            label = root.split('/') [-1]
            docs = []
            t = trange(len(files))
            for i in t:
                txt = files[i]
                doc = []
                txt_path = os.path.join(root, txt)
                with open(txt_path, 'r', encoding='gb18030', errors='ignore') as ft:
                    lines = ft.readlines()
                    for line in lines:
                        sentence = line.strip()
                        sentence = line.replace(' ', '')                    # 去除空格符
                        sentence = str_filter(sentence)
                        tokens = fenci.bi_direction_matching(sentence)      # 中文分词：双向最大匹配算法
                        texts = ' '.join(tokens)
                        if texts != '':
                            doc.append(texts)
                            corpus.append(texts)
                if len(doc) > 0:
                    docs.append(doc)
            dict_train[label] = docs
    
    with open('./data/train.json', 'w', encoding='utf-8') as fj:            # 保存为 json 文件
        json.dump(dict_train, fj, ensure_ascii=False)
    print('Training dataset is successfully built...')


def build_test_dataset(fenci):
    '''
    构建测试数据集
    '''
    print('Start to build testing dataset...')
    dict_test  = {}

    test_files = os.listdir(original_test_path)
    t = trange(len(test_files))
    for i in t:
        txt = test_files[i]
        doc = []
        index = txt.split('.')[0]
        txt_path = os.path.join(original_test_path, txt)
        with open(txt_path, 'r', encoding='gb18030', errors='ignore') as ft:
            lines = ft.readlines()
            for line in lines:
                sentence = line.strip()
                sentence = line.replace(' ', '')                              # 去除空格符
                sentence = str_filter(sentence)
                tokens = fenci.bi_direction_matching(sentence)                # 中文分词
                texts = ' '.join(tokens)
                if texts != '':
                    doc.append(texts)
                    corpus.append(texts)
        dict_test[index] = [doc]

    with open('./data/test.json', 'w', encoding='utf-8') as fj:
        json.dump(dict_test, fj, ensure_ascii=False)
    print('Testing dataset is successfully built...')


if __name__ == "__main__":
    word_set = set()                        # 使用集合类型作为查找字典，可以大大提高效率
    stop_words = set()
    with open('./data/word_list.txt', 'r', encoding='gb18030') as f_w:
        lines = f_w.readlines()
        for line in lines:
            word = line.strip().split()
            for w in word:
                word_set.add(w)

    with open('./data/stop_words.txt', 'r', encoding='gb18030') as f_s:
        lines = f_s.readlines()
        for line in lines:
            word = line.strip().split()
            for w in word:
                stop_words.add(w)

    fenci = Segment_Words(stop_words, word_set)     # 初始化分词工具
    
    print('-' * 100)

    build_train_dataset(fenci)
    build_test_dataset(fenci)

    # 基于所有文本数据构建该任务的语料库
    with open('./data/corpus.txt', 'w', encoding='utf-8') as ft:
        for line in corpus:
            ft.write(line)
            ft.write('\n')
    print('Build dataset is completed...')

    # 基于该任务的全体语料库训练词向量
    corpus = word2vec.Text8Corpus(corpus_path)
    model = word2vec.Word2Vec(corpus, sg=1, size=300, window=5, min_count=1, workers=4)
    model.save(save_path) 

    print('Build word_to_vector is completed...')
