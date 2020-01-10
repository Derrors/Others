<div align=center>
    <div style="font-size:24px">
        <b>文本分类</b>
    </div>
</div>

## Requirement

    * Python 3.7.4
    * torch 1.2.0
    * tqdm
    * gensim
    * 运行环境：1080 Ti GPU CUDA 10.0

## Prepare

1. 安装以上需要的所有模块.
2. 数据预处理:   
    * 数据清洗，去除所有非中文字符;
    * 中文分词，使用自己编写实现的基于双向最大匹配算法的分词工具;
    * 构建语料库，将数据集分词后构建全体文本的语料库;
    * 预训练词向量，基于全体语料库使用 gensim 训练本任务的词向量(维度：300);
    ```bash
    python build_dataset.py
    ```
    注：准备工作仅需要完成一次即可.   

3. 数据预处理完成后，项目目录结构如下：

```python
├── original_data                               # 原始数据集
│   ├── train
│   └── test
│
├── data                                        # 预处理后的数据文件
│   ├── corpus.txt                              # 全体语料
│   ├── stop_words.txt                          # 停用词
│   ├── word_list.txt                           # 词表
│   ├── train.json                              # 格式化后的训练集
│   ├── test.json                               # 格式化后的测试集
│   ├── word2vec.bin                            # 预训练的词向量
│   ├── word2vec.bin.wv.vectors.npy
│   └── word2vec.bin.trainables.syn1neg.npy
│
├── saved_model                                 # 最优模型参数
│   └── text_rcnn.pth
│
├── others                                      # 相关提交文件
│   ├── result.txt                              # 训练过程输出
│   └── Result.png                              # 训练结果图示
│
├── fenci.py                # 中文分词
├── text_rcnn.py            # TextRCNN 模型
├── build_dataset.py        # 数据预处理及构建数据集
├── data_loader.py          # 数据加载器
├── train_and_test.py       # 模型训练和测试
└── run_model.py            # 复现最终结果
```

## Train

* 重新训练文本分类器，依次运行：
    ```bash
    python build_dataset.py
    python train_and_test.py
    ```

## Reimplement the Result

* 对最终实现的分类准确率 90.04% 进行复现：
    ```bash
    # 无需 Prepare、Train 阶段, 直接运行：
    python run_model.py
    # 确保 saved_model 目录下有保存的参数文件
    # 模型参数是在 GPU 上训练得到，因此需在 GPU 上加载模型及参数进行复现
    ```

## Result

<div align=center>
    <img src="./others/Result.png" width="500px">
</div>
