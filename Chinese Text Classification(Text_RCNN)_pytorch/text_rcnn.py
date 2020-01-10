# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/05

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, embedding_size, output_dim, hidden_size, num_layers, bidirectional, dropout):
        super(TextRCNN, self).__init__()

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.linear = nn.Linear(2 * hidden_size + embedding_size, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            输入维度：[batch size, seq_length, embedding_size]
        '''
        outputs, _ = self.rnn(x)                                          # 双向 RNN
        # outputs: [batch_size, seq_length, hidden_size * 2]

        outputs = self.dropout(outputs)

        x = torch.cat((outputs, x), 2)                                    # 将 RNN 的输出与 x 拼接
        # x: [batch_size, seq_length, embedding_size + hidden_size * 2]

        outputs = torch.tanh(self.linear(x)).permute(0, 2, 1)             # 前向神经网路、激活
        # outputs: [batch_size, hidden_size * 2, seq_length]

        outputs = F.max_pool1d(outputs, outputs.size()[2]).squeeze(2)     # 最大池化
        # outputs: [batch_size, hidden_size * 2]

        return self.fc(outputs)                                           # 全连接