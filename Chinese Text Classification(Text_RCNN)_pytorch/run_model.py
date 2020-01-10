# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/29

import torch
import numpy as np 
from tqdm import trange
from text_rcnn import TextRCNN
from data_loader import Dataloader
from train_and_test import evaluate


def get_the_final_result():
    # 参数配置
    batch_size = 512
    seq_length = 20
    embeddings_size = 300
    hidden_size = 256
    num_layers = 2
    num_classes = 9
    learning_rate = 0.003
    dropout = 0.3
    
    # 数据文件路径
    word2vec_path = './data/word2vec.bin'
    train_file = './data/train.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型
    model = TextRCNN(embeddings_size, num_classes, hidden_size, num_layers, True, dropout)
    model.to(device)

    # 加载训练好的模型参数
    checkpoints = torch.load('./saved_model/text_rcnn.pth')
    model.load_state_dict(checkpoints['model_state'])

    # 加载数据
    data_loader = Dataloader(word2vec_path, batch_size, embeddings_size, seq_length, device)        # 初始化数据迭代器
    texts, labels = data_loader.load_data(train_file, shuffle=True, mode='train')                   # 加载数据
    print('Data load completed...')

    # 在测试集上进行测试
    test_texts   = texts[int(len(texts) * 0.8): ]
    test_labels  = labels[int(len(texts) * 0.8): ]
    steps = len(test_texts) // batch_size
    loader = data_loader.data_iterator(test_texts, test_labels)

    # 测试集上的准确率
    accuracy = evaluate(model, loader, steps)
    print('The final result(Accuracy in Test) is %.2f' % (accuracy * 100))


if __name__ == "__main__":
    get_the_final_result()

