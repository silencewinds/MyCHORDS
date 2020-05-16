'''
编码器解码器模型

silence   2019.12.14
'''

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

MAXLEN=10

#编码器
class Encoder_RNN(nn.Module):
    def __init__(self,input_size,embed_dim,hidden_size,layers_num=1):
        super(Encoder_RNN, self).__init__()
        self.embed_dim=embed_dim
        self.hidden_size=hidden_size
        self.layers_num=layers_num
        self.embedding=nn.Embedding(input_size,embed_dim)    #input_size为字典单词数,embed_dim为嵌入维度,会给每个单词按照词向量维度随机初始化
        self.gru=nn.RNN(embed_dim,hidden_size)               #GRU单元输入是一个单词（词向量维度表示），输出对应隐藏层神经元个数
    def forward(self,input,hidden):
        input=input.unsqueeze(1)
        embedded=self.embedding(input)
        output=embedded.permute(1,0,2)
        for i in range(self.layers_num):                     #这里的层数与RNN函数中的层数参数不同，函数中的是定义在神经网络中的，有多少层就有多少个隐藏状态；而此处层数是定义在网络结构外面的，始终只有一个隐藏状态
            output,hidden=self.gru(output,hidden)
        return output,hidden
    def initHidden(self):                                    #初始隐藏层状态，一般赋权为零
        result=Variable(torch.zeros(1,1,self.hidden_size))   #层数为1，批次为1
        return result

#普通解码器
class Decoder_RNN(nn.Module):
    def __init__(self,hidden_size,embed_dim,output_size, n_layers=1):
        super(Decoder_RNN, self).__init__()
        self.embed_dim=embed_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_dim)
        self.gru = nn.RNN(embed_dim, hidden_size)
        #增加一个线性层将维度转为单词数目，然后通过softmax分类
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    def forward(self, input, hidden):
        output = self.embedding(input)    # batch, 1, hidden
        output = output.permute(1, 0, 2)  # 1, batch, hidden
        for i in range(self.n_layers):
            if i==1:                                            #在最后一层将输出hs256转为128em大小
                yyy = nn.Linear(128, 5)
                output = yyy(output)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.embed_dim))
        return result

#注意力机制解码器
class AttentionDecoder_RNN(nn.Module):
    def __init__(self,hidden_size,embed_dim,output_size,layers_num,dropout=0.5,max_len=MAXLEN):
        super(AttentionDecoder_RNN, self).__init__()
        self.embed_dim=embed_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers_num = layers_num
        self.dropout= dropout
        self.max_length = max_len

        self.embedding = nn.Embedding(self.output_size, self.embed_dim)
        #将要输入的词向量和隐藏状态拼在一起，接着通过线性层加 softmax 激活层输出固定长度的序列,这个序列就是注意力序列，每个数就是这个单词在Ci中的注意力系数
        #在此处匹配函数通过一层神经网络实现
        self.attn = nn.Linear(self.hidden_size+self.embed_dim, self.max_length)
        #将根据该生成词得到的Ci中间语义和输入词通过线性层组合
        self.attn_combine = nn.Linear(self.hidden_size+self.embed_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.RNN(self.embed_dim, self.hidden_size)
        #输出，经过线性层将维度从神经元个数转为输出维度
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.trans=nn.Linear(self.hidden_size,self.embed_dim)
    def forward(self,input,hidden,en_outputs):
        '''
        input: batch, 1
        hidden: 1, batch, hidden
        encoder_outputs: length, hidden
        '''
        embedded=self.embedding(input)                                  #batch,1,hidden
        emdedded=self.dropout(embedded)
        embedded=embedded.squeeze(1)
        attn_w=F.softmax(self.attn(torch.cat((embedded,hidden[0]),1)),dim=1)  #将词向量与隐藏状态拼接一起通过线性层和softmax得到一个固定长度的概率序列，作为注意力概率参数。得到 batch,max_len
        en_outputs = en_outputs.unsqueeze(0)                            #batch,max_len,hidden
        attn_applied = torch.bmm(attn_w.unsqueeze(1), en_outputs)       #将编码过程的输出与注意力权重做矩阵乘法（加权求和法）得到Ci     batch,1,hidden
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)      #将输入拼接，通过一个线性层转化为网络接受的维度
        output = self.attn_combine(output).unsqueeze(0)                 #1，batch,hidden
        for i in range(self.layers_num):
            output=self.trans(output)
            output = F.relu(output)
            # print(hidden[0].dtype)
            # hidden[0]=hidden[0].unsqueeze(0)
            # print(hidden[0].size())
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output.squeeze(0)),dim=1)
        return output, hidden, attn_w
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.embed_dim))
        return result