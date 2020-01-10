# -*- coding: utf-8 -*-
# author: Qinghe Li
# date: 2019/10/09

import re

def convert_to_tokens_and_labels(source_file, sentences_file, labels_file):
    ts = open(sentences_file, 'w')
    tl = open(labels_file, 'w')

    with open(source_file, 'r', encoding='gb18030') as sf:
        lines = sf.readlines()
        for line in lines:
            sentence = re.sub('  ', '', line).strip()
            for  i in range(len(sentence)):
                ts.write(sentence[i] + ' ')

            tokens = line.strip().split()
            for token in tokens:
                if len(token) == 1:
                    tl.write('S ')
                elif len(token) == 2:
                    tl.write('B E ')
                else:
                    M_num = len(token) - 2
                    tl.write('B ')
                    tl.write('M ' * M_num)
                    tl.write('E ')   
            ts.write('\n')
            tl.write('\n')
    ts.close()
    tl.close()
            
    print('Data preprocessing completed --------------------')


if __name__ == "__main__":
    train_text = './data/pku_train.txt'
    test_text  = './data/pku_test.txt'
    convert_to_tokens_and_labels(train_text, './data/train/sentences.txt', './data/train/labels.txt')
    convert_to_tokens_and_labels(test_text, './data/test/sentences.txt', './data/test/labels.txt')
