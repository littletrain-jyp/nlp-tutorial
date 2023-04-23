# -*- coding: utf-8 -*-
# @Time         : 2023/4/10 16:25
# @Author       : Yupeng Ji
# @File         : Word2Vec-Skipgram(Softmax).py
# @Description  : 动手实现Word2Vec

"""
NNLM 主要是通过前n个词预测后一个词，即给定上文预测下文，词向量是副产物
Word2Vec 中的skip-gram是给定该词预测上下文，cbow是给定上下文预测该词，因为Word2Vec主要就是要词向量，所以它可以随性训练。

该文件仅实现了最简单的skip-gram softmax，实际上还有很多优化技巧，比如负例采样。
负例采样 其实是 把神经网络模型改为了逻辑回归模型，把input和target都作为输入，label是0或1代表是否是相邻的。这样就要针对性的增加一些负采样了，
不然模型就学会将他们全变为1.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# skip-gram model
class Word2Vec_skipgram(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(Word2Vec_skipgram, self).__init__()
        self.voc_size = voc_size
        self.emb_size = emb_size
        # weight
        self.W = nn.Linear(self.voc_size, self.emb_size, bias=False)
        # voc_size weight
        self.WT = nn.Linear(self.emb_size, self.voc_size, bias=False)

    def forward(self, X):
        #X:[batch_size, voc_size]
        hidden_layer = self.W(X) # hidden_layer:[batch_size, emb_size]
        output = self.WT(hidden_layer) # output: [batch_size, voc_size]
        return output

if __name__ == "__main__":
    batch_size = 2
    emb_size = 2

    sentences = ['苹果 香蕉 水果', '橘子 桃子 水果', '猫 狗 动物', '水果 苹果 橘子' ]
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_list = ' '.join(sentences).split()
    word_sequence = ' '.join(sentences).split()
    word_list = list(set(word_list))

    word2id = {w: i for i, w in enumerate(word_list)}
    id2word = {i: w for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    model = Word2Vec_skipgram(voc_size, emb_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Make skip gram of one size window
    skip_grams = []

    for i in range(1, len(word_sequence) - 1):
        target = word2id[word_sequence[i]]
        context = [word2id[word_sequence[i-1]], word2id[word_sequence[i+1]]]
        for w in context:
            skip_grams.append([target, w]) # 实际上skip-gram


    # 打乱随机batch
    def random_batch():
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

        for i in random_index:
            random_inputs.append(np.eye(voc_size)[skip_grams[i][0]]) #target
            random_labels.append(skip_grams[i][1]) #context word

        return random_inputs, random_labels

    # training
    for epoch in range(5000):
        input_batch, output_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(output_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch:{epoch}, loss:{loss}")

    print(model.parameters())
    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
