# -*- coding: utf-8 -*-
# @Time         : 2023/4/10 13:25
# @Author       : Yupeng Ji
# @File         : NNLM.py
# @Description  : 动手实现NNLM

import torch
import torch.nn as nn
import torch.optim as optim

class NNLM(nn.Module):
    """Neural Network language Model architecture
    Args:
        n_class: 词表大小
        emb_size: 词向量维度
        n_step: 类似于n-gram， 即用前n个词预测后一个词
        n_hidden：隐藏层神经元数量
    """
    def __init__(self, n_class, emb_size, n_step, n_hidden):
        super(NNLM, self).__init__()

        self.n_class = n_class
        self.emb_size = emb_size
        self.n_step =n_step
        self.n_hidden = n_hidden

        self.C = nn.Embedding(self.n_class, self.emb_size)
        self.H = nn.Linear(self.n_step * self.emb_size, self.n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(self.n_hidden))
        self.U = nn.Linear(self.n_hidden, self.n_class, bias=False)
        self.W = nn.Linear(self.n_step * self.emb_size, self.n_class, bias=False) # 直接连接输入层与输出层
        self.b= nn.Parameter(torch.ones(self.n_class))

    def forward(self,X):
        X = self.C(X) # X:[batch_size, n_step, m]
        X = X.view(-1, self.n_step * self.emb_size) # X:[batch_size, n_step * m]  拼接词向量
        tanh = torch.tanh(self.H(X) + self.d) # tanh: [batch_size, n_hidden]
        output = self.b + self.U(tanh) + self.W(X) #output: [batch_size, n_class]
        return output

if __name__ == '__main__':

    sentences = ['我 喜欢 打球', '我 研究 自然语言处理', '我 在 吃饭']
    word_list = ' '.join(sentences).split()
    word_set = list(set(word_list))

    word2id = {w: i for i, w in enumerate(word_set)}
    id2word = {i: w for i, w in enumerate(word_set)}

    print(sentences)

    def make_batch(sentences):
        input_batch = []
        output_batch = []

        for sen in sentences:
            word = sen.split()
            input = [word2id[i] for i in word[:-1]]
            target = word2id[word[-1]]

            input_batch.append(input)
            output_batch.append(target)
        return input_batch, output_batch

    input_batch, output_batch = make_batch(sentences)
    input_batch = torch.LongTensor(input_batch)
    output_batch = torch.LongTensor(output_batch)

    n_class = len(word2id)
    emb_size = 3
    n_step = 2
    n_hidden = 2

    nnlm_model = NNLM(n_class, emb_size, n_step, n_hidden)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nnlm_model.parameters(), lr=0.001)

    # 迭代训练
    for i in range(2000):
        optimizer.zero_grad() # 梯度清零,就是把loss关于weight的导数变为0

        output = nnlm_model(input_batch)
        loss = criterion(output, output_batch)
        loss.backward()
        optimizer.step() #只有调用该方法，模型才会更新参数
        if i % 100 == 0:
            print(f"step:{i}, loss:{loss}")

    # 预测
    predict = nnlm_model(input_batch).data.max(1, keepdim =True)[1]

    print([sentence.split()[:2] for sentence in sentences], "---->",
          [id2word[n.item()] for n in predict.squeeze()]) # squeeze()表示把数组中维度为1的维度去掉，对张量的维度进行减少操作


# reference:
#   1.https://blog.csdn.net/manzubaolong/article/details/109131663
#   2.https://blog.csdn.net/weixin_50706330/article/details/127708430?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-127708430-blog-109131663.235%5Ev28%5Epc_relevant_t0_download&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-127708430-blog-109131663.235%5Ev28%5Epc_relevant_t0_download&utm_relevant_index=7

