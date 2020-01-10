# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/07

import random
import numpy as np 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from data_loader import Dataloader
from text_rcnn import TextRCNN

# 配置文件路径
word2vec_path = './data/word2vec.bin'
train_file = './data/train.json'
vaild_file = './data/test.json'


def train(model, data_iterator, train_steps, loss_fn, optimizer, scheduler):
    '''
    模型训练
    '''
    model.train()
    
    t = trange(train_steps)
    for i in t:
        model.zero_grad()                                                               # 梯度清零

        batch_embeddings, batch_labels = next(data_iterator)
        output = model(batch_embeddings)
        loss = loss_fn(output, batch_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=100)           # 梯度裁剪
        optimizer.step()

        t.set_postfix(loss='{:05.6f}'.format(loss))
    scheduler.step()


def evaluate(model, data_iterator, steps):
    '''
    模型评估：计算预测的准确率
    '''
    model.eval()

    true = []
    pred = []

    for _ in range(steps):
        batch_embeddings, batch_labels = next(data_iterator)
        output = model(batch_embeddings)

        output = output.detach().to('cpu').numpy()
        output = np.argmax(output, 1)
        batch_labels = batch_labels.to('cpu').numpy()

        true.extend(batch_labels)
        pred.extend(output)
    assert len(true) == len(pred)
    
    pred = np.array(pred, dtype=int)
    true = np.array(true, dtype=int)
    accuracy = np.sum(pred == true) / len(pred)

    return accuracy

def train_and_test(model, optimizer, criterian, scheduler, batch_size, embeddings_size, seq_length, save_madel_path, epochs, device):
    '''
    配置模型的训练和测试过程
    '''
    data_loader = Dataloader(word2vec_path, batch_size, embeddings_size, seq_length, device)        # 初始化数据迭代器
    texts, labels = data_loader.load_data(train_file, shuffle=True, mode='train')                   # 加载数据
    print('Data load completed...')

    # 将数据集划分为训练集和测试集
    train_texts  = texts[: int(len(texts) * 0.8)]       
    test_texts   = texts[int(len(texts) * 0.8): ]
    train_labels = labels[: int(len(texts) * 0.8)]
    test_labels  = labels[int(len(texts) * 0.8): ]
    
    # 获取训练/测试步数
    train_steps = len(train_texts) // batch_size
    test_steps = len(test_texts) // batch_size

    best_test_acc = 0.0                             # 记录训练过程中的最优结果

    for e in range(1, epochs + 1):
        print('Epoch {}/{}'.format(e, epochs))

        # 模型训练
        train_data_iterator = data_loader.data_iterator(train_texts, train_labels)
        train(model, train_data_iterator, train_steps, criterian, optimizer, scheduler)

        train_data_iterator = data_loader.data_iterator(train_texts, train_labels)
        test_data_iterator = data_loader.data_iterator(test_texts, test_labels)
        
        # 模型测试
        train_accuracy = evaluate(model, train_data_iterator, train_steps)
        test_accuracy = evaluate(model, test_data_iterator, test_steps)
        print('Training accuracy: ', train_accuracy)
        print('Testing accuracy: ', test_accuracy)

        improve_acc = test_accuracy - best_test_acc
        if improve_acc > 0:                                     # 保存最优模型的参数
            print('Found a new best accuracy...')
            best_test_acc = test_accuracy
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e,
                'accuracy': best_test_acc
            }
            torch.save(checkpoint, save_madel_path)


if __name__ == "__main__":
    # 参数配置
    epochs = 50
    batch_size = 512
    seq_length = 20
    embeddings_size = 300
    hidden_size = 256
    num_layers = 2
    num_classes = 9
    learning_rate = 0.003
    dropout = 0.3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置随机种子
    random.seed(2020)
    torch.manual_seed(2020)
    
    # 加载文本分类模型 TextRCNN
    model = TextRCNN(embeddings_size, num_classes, hidden_size, num_layers, True, dropout)
    model.to(device)

    # 定义损失函数和优化器
    criterian = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05 * epoch))
    
    print('-' * 100)
    train_and_test(model, optimizer, criterian, scheduler, batch_size, embeddings_size, seq_length, './saved_model/text_rcnn.pth', epochs, device)
