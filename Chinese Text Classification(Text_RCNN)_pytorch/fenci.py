# -*- coding: utf-8 -*-
# code by Qinghe Li
# date: 2019/11/07

class Segment_Words(object):
    def __init__(self, stop_words, words_set):
        '''
            初始化函数
            words_set: 词典
        '''
        self.stop_words = stop_words
        self.words_set = words_set
        self.max_length = self.set_max_word_length()        # 词典中的最长词的长度
    
    def set_max_word_length(self):
        '''
            统计词典中最长词的长度
        '''
        count = [len(word) for word in self.words_set]
        return max(count)

    def forward_maximum_matching(self, sentence):
        '''
            前向最大匹配算法
        '''                         
        words = []
        start, end = 0, 0                               # start, end指示分词起止位置
        while start < len(sentence):
            if end + self.max_length <= len(sentence):
                end = start + self.max_length
            else:
                end = len(sentence)
            match_size = len(sentence[start: end])      # match_size： 原始匹配窗口大小
            for i in range(match_size):
                word = sentence[start: end-i]
                if word in self.words_set or i == match_size - 1:      # 若词不在词典中，则减去最后一个字符进行匹配
                    if word not in self.stop_words:
                        words.append(word)
                    start += match_size - i             # 匹配窗口根据当前匹配结果进行移动
                    break
        return words

    def backward_maximum_matching(self, sentence):
        '''
            逆向最大匹配算法
        '''                     
        words = []
        start, end = len(sentence), len(sentence)
        while start > 0:
            if end - self.max_length >= 0:
                end = start - self.max_length
            else:
                end = 0
            match_size = len(sentence[end: start])
            for i in range(match_size):
                word = sentence[end+i: start]
                if word in self.words_set or i == match_size - 1:
                    if word not in self.stop_words:
                        words.append(word)
                    start -= match_size - i
                    break
        words.reverse()
        return words

    def bi_direction_matching(self, sentence):
        '''
            双向最大匹配算法
        '''
        fmm = self.forward_maximum_matching(sentence)          # 依次执行前向最大匹配、逆向最大匹配
        bmm = self.backward_maximum_matching(sentence)         # 对比两种结果择优选择

        if len(fmm) > len(bmm):                       # 首先考虑分词数目，取词数较少的结果
            return bmm
        elif len(fmm) < len(bmm):                     # 分词数目相同时，取单字数目较少的结果
            return fmm
        elif fmm == bmm:
                return fmm
        else:
            count_fmm = [len(s) for s in fmm].count(1)
            count_bmm = [len(s) for s in bmm].count(1)
            if count_fmm > count_bmm:
                return bmm
            else:
                return fmm
