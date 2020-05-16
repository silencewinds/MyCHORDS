'''
投入数据，训练模型，保存模型参数

silence   2019.12.14
'''

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from 旋律2和弦.process_data import MyDataset
from 旋律2和弦.model.EnDe_model import AttentionDecoder_RNN,Decoder_RNN,Encoder_RNN

St_tag=0
Ed_tag=1
MAX_LEN=10
language_dataset=MyDataset()
language_dataloader=DataLoader(language_dataset,shuffle=True)

input_size=language_dataset.input_lang_words
output_size=language_dataset.output_lang_words
#print(input_size)
#print(output_size)
#隐藏层神经元个数
hidden_size=128
#词向量特征维度
embed_dim=56
EPOCH=30

encoder=Encoder_RNN(input_size,embed_dim,hidden_size)
decoder=Decoder_RNN(hidden_size,embed_dim,output_size,2)
attn_decoder=AttentionDecoder_RNN(hidden_size,embed_dim,output_size,2)
use_attention=True

#做损失函数变化图像
def showPlot(points):
    plt.figure()
    x = np.arange(len(points))
    plt.plot(x, points)
    plt.show()

def train(encoder,decoder,epoch,use_attention):
    param=list(encoder.parameters())+list(decoder.parameters())
    optimizer=optim.Adam(param,lr=1e-3)
    criterion=nn.NLLLoss()
    plot_losses=[]
    for epoch in range(epoch):
        since=time.time()
        run_loss=0
        printloss_total=0
        totalloss=0
        for i,data in enumerate(language_dataloader):

            in_lang,out_lang=data
            in_lang = Variable(in_lang)
            out_lang = Variable(out_lang)

            #初始化编码器
            encoder_outputs=Variable(torch.zeros(MAX_LEN,encoder.hidden_size))
            encoder_hidden=encoder.initHidden()
            for j in range(in_lang.size(1)):
                encoder_output,encoder_hidden=encoder(in_lang[:,j],encoder_hidden)
                encoder_outputs[j]=encoder_output[0][0]
            #初始化解码器
            decoder_input=Variable(torch.LongTensor([[St_tag]]))
            decoder_hidden=encoder_hidden

            loss=0
            if use_attention:
                for k in range(out_lang.size(1)):
                    decoder_output,decoder_hidden,decoder_attention=attn_decoder(decoder_input,decoder_hidden,encoder_outputs)
                    loss+=criterion(decoder_output,out_lang[:,k])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]
                    decoder_input=Variable(torch.LongTensor([[ni]]))
                    if ni==Ed_tag:       #如果生成了结束标志则翻译完成，停止
                        break
            else:
                #不使用注意力机制，每次生成翻译词都使用同一个状态C
                for k in range(out_lang.size(1)):
                    decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
                    loss += criterion(decoder_output, out_lang[:, k])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]
                    decoder_input=Variable(torch.LongTensor([[ni]]))
                    if ni==Ed_tag:
                        break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()             #通过优化器来优化参数w

            run_loss += loss.data.item()
            printloss_total += loss.data.item()
            totalloss += loss.data.item()

            # if (i + 1) % 5000 == 0:
            #     print('{}/{}, 损失值:{:.6f}'.format(i + 1, len(language_dataloader), run_loss / 5000))
            #     run_loss = 0
            if (i + 1) % 100 == 0:
                plot_loss = printloss_total / 100
                plot_losses.append(plot_loss)
                printloss_total = 0
        during = time.time() - since
        print('完成 {}/{} , 损失值:{:.6f}, 用时:{:.0f}s'.format(epoch + 1, EPOCH, totalloss / len(language_dataset), during))
        print()
    showPlot(plot_losses)


if use_attention:
    print("==>traing......")
    train(encoder,attn_decoder,EPOCH,True)
else:
    print("==>traing......")
    train(encoder,decoder,EPOCH,False)
print("Finish training!")

#保存模型参数
if use_attention:
    torch.save(encoder.state_dict(), './encoder_param.pkl')
    torch.save(attn_decoder.state_dict(), './attentiondecoder_param.pkl')
else:
    torch.save(encoder.state_dict(), './encoder_param.pkl')
    torch.save(decoder.state_dict(), './decoder_param.pkl')