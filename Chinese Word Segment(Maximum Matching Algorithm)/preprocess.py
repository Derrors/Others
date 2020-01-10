# -*- coding: utf-8 -*-
# author: Qinghe Li
# date: 2019/09/30

import re

# 配置文件路径
stop_words_path     = './data/stop_words.txt'
words_set_path     = './data/words_list.txt'
test_sentences_path = './data/test_sentences.txt'
sentences_path      = './data/sentences.txt'

def load_word_data():
    '''
        加载数据文件
    '''
    stop_words = set()         # 停用词表
    words_set = set()       # 词典

    with open(stop_words_path, 'r', encoding='gb18030') as f_s:
        lines = f_s.readlines()
        for line in lines:
            word = line.strip().split()
            for w in word:
                stop_words.add(w)

    with open(words_set_path, 'r', encoding='gb18030') as f_w:
        lines = f_w.readlines()
        for line in lines:
            word = line.strip().split()
            for w in word:
                words_set.add(w)

    return stop_words, words_set

def data_preprocess():
    '''
        对原始语料进行数据预处理
        返回测试语料及对应的标准分词结果

    '''
    chars = re.compile(r'[^\u4e00-\u9fa5]')         # 非中文字符
    f_s = open(test_sentences_path, 'w')

    stop_words, _ = load_word_data()
    sentences = []
    labels = []
    
    with open(sentences_path, 'r', encoding='gb18030') as f:
        lines = f.readlines()
        for line in lines:
            test_sentence = re.sub(chars, '', line)  # 去除所有非中文字符
            sentences.append(test_sentence)
            f_s.write(test_sentence + '\n')

            test_label = re.sub(chars, ' ', line).strip().split()
            label = []
            for i in range(len(test_label)):
                if test_label[i] not in stop_words:
                    label.append(test_label[i])
            labels.append(label)      
    f_s.close()
    print('Data preprocessing completed ------------------')
    return sentences, labels
