# -*- coding: utf-8 -*-
# author: Qinghe Li
# date: 2019/10/02

# 导入数预处理模块
import preprocess as pre
from tqdm import trange


class Segment_Words(object):
    def __init__(self, stop_words, words_set, labels):
        '''
            初始化函数
            stop_words: 停用词表
            words_set: 词典
            labels    : 测试语料的标准分词结果
        '''
        self.stop_words = stop_words
        self.words_set = words_set
        self.labels = labels
        self.max_length = self.set_max_word_length()        # 词典中的最长词的长度

    def set_max_word_length(self):
        '''
            统计词典中最长词的长度
        '''
        count = [len(word) for word in self.words_set]
        return max(count)

    def result_evalutate(self, result):
        '''
            分词结果评估
            Precision : 精确率
            Recall    : 召回率
            F1-Measure: 综合性能指标
        '''
        r_s, r_t, l_s = 0, 0, 0                             # r_s, r_t, l_s 分别为分词结果中词的总数、分词结果中正确分词的总数、
        # 标准分词结果中词的总数
        for i in range(len(result)):
            r_s += len(result[i])
            l_s += len(self.labels[i])
            for word in result[i]:
                if word in self.labels[i]:
                    r_t += 1

        precision = r_t / r_s
        recall = r_t / l_s
        f1_measure = 2 * precision * recall / (precision + recall)
        print('Precision: %.4f' % (precision), 'Recall: %.4f' % (recall), 'F1_Measure: %.4f' % (f1_measure), sep=' 丨 ')
        return

    def write_to_txt(self, result, txt_name):
        '''
            分词结果写入txt文件
        '''
        file_name = str('./data/' + txt_name)
        with open(file_name, 'w') as fw:
            for line in result:
                for word in line:
                    fw.write(word + ' ')
                fw.write('\n')
        return

    def forward_maximum_matching(self, sentences):
        '''
            前向最大匹配算法
        '''
        print('Forward Maximum Matching is starting ----------')

        result = []
        t_f = trange(len(sentences))
        for t in t_f:                                       # 对每条测试语料进行分词
            sentence = sentences[t]
            words = []
            start, end = 0, 0                               # start, end指示分词起止位置
            while start < len(sentence):
                if end + self.max_length <= len(sentence):
                    end = start + self.max_length
                else:
                    end = len(sentence)
                match_size = len(sentence[start: end])      # match_size： 原始匹配窗口大小
                for i in range(match_size):
                    word = sentence[start: end - i]
                    if word in self.words_set or i == match_size - 1:      # 若词不在词典中，则减去最后一个字符进行匹配
                        if word not in self.stop_words:
                            words.append(word)
                        start += match_size - i             # 匹配窗口根据当前匹配结果进行移动
                        break
            result.append(words)

        self.result_evalutate(result)                       # 评估分词结果
        self.write_to_txt(result, 'result_fmm.txt')         # 将分词结果写入文件
        print('Forward Maximum Matching completed ----------')
        return result

    def backward_maximum_matching(self, sentences):
        '''
            逆向最大匹配算法
        '''
        print('Backward Maximum Matching is starting ----------')

        result = []
        t_b = trange(len(sentences))
        for t in t_b:                                        # 流程与前向最大匹配算法相似，只是匹配窗口从后向前移动
            sentence = sentences[t]
            words = []
            start, end = len(sentence), len(sentence)
            while start > 0:
                if end - self.max_length >= 0:
                    end = start - self.max_length
                else:
                    end = 0
                match_size = len(sentence[end: start])
                for i in range(match_size):
                    word = sentence[end + i: start]
                    if word in self.words_set or i == match_size - 1:
                        if word not in self.stop_words:
                            words.append(word)
                        start -= match_size - i
                        break
            words.reverse()
            result.append(words)

        self.result_evalutate(result)
        self.write_to_txt(result, 'result_bmm.txt')
        print('Backward Maximum Matching completed ----------')
        return result

    def bi_direction_matching(self, sentences):
        '''
            双向最大匹配算法
        '''
        fmm = self.forward_maximum_matching(sentences)          # 依次执行前向最大匹配、逆向最大匹配
        bmm = self.backward_maximum_matching(sentences)         # 对比两种结果择优选择

        print('Bi-direction Matching is starting ----------')

        result = []
        t = trange(len(sentences))
        for i in t:
            if len(fmm[i]) > len(bmm[i]):                       # 首先考虑分词数目，取词数较少的结果
                result.append(bmm[i])
            elif len(fmm[i]) < len(bmm[i]):                     # 分词数目相同时，取单字数目较少的结果
                result.append(fmm[i])
            elif fmm[i] == bmm[i]:
                result.append(fmm[i])
            else:
                count_fmm = [len(s) for s in fmm[i]].count(1)
                count_bmm = [len(s) for s in bmm[i]].count(1)
                if count_fmm > count_bmm:
                    result.append(bmm[i])
                else:
                    result.append(fmm[i])

        self.result_evalutate(result)
        self.write_to_txt(result, 'result_bm.txt')
        print('Bi-direction Matching completed ----------')
        return result


if __name__ == "__main__":
    stop_words, words_set = pre.load_word_data()
    sentences, labels = pre.data_preprocess()
    fenci = Segment_Words(stop_words, words_set, labels)
    result = fenci.bi_direction_matching(sentences)
