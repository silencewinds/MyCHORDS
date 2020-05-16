'''
模型评价：
1.随机选取数据集中样本进行翻译。
2.随机使用数据集外的样本翻译，查看泛化能力。我使用了《走马》这首歌的四节主旋律，得到其相对应的四节编配和弦（字符串形式，若要播放请运行play文件）

silence    2019.12.14
'''

import random
import torch
from torch.autograd import Variable
from 旋律2和弦.process_data import MyDataset
from 旋律2和弦.model.EnDe_model import AttentionDecoder_RNN, Decoder_RNN, Encoder_RNN

St_tag = 0
Ed_tag = 1
MAXLEN = 10
use_attention = True
lang_dataset = MyDataset()
print("123")


# 模型翻译
def evaluate(encoder, decoder, in_lang, max_length=MAXLEN):
    input_variable = Variable(in_lang)
    input_variable = input_variable.unsqueeze(0)
    input_length = input_variable.size(1)
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[:, i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[St_tag]]))
        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(max_length, max_length)
        decoded_words = []

    if use_attention:
        for j in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[j] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == Ed_tag:
                decoded_words.append('<结束>')
                break
            else:
                # 将得到的数值通过字典转化为相应的单词
                decoded_words.append(lang_dataset.output_lang.index2word[ni.item()])
            decoder_input = Variable(torch.LongTensor([[ni]]))  # 更新输入
    else:
        for j in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == Ed_tag:
                decoded_words.append('<结束>')
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni.item()])
            decoder_input = Variable(torch.LongTensor([[ni]]))
    if use_attention:
        return decoded_words, decoder_attentions[:j + 1]
    else:
        return decoded_words


# 随机选择10句进行翻译观察效果
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair_idx = random.choice(list(range(len(lang_dataset))))
        pair = lang_dataset.pairs[pair_idx]
        in_lang, out_lang = lang_dataset[pair_idx]
        print('待翻译：', pair[0])
        print('应译为：', pair[1])
        if use_attention:
            output_words, attentions = evaluate(encoder, decoder, in_lang)
        else:
            output_words = evaluate(encoder, decoder, in_lang)
        output_sentence = ' '.join(output_words)
        print('机器翻译为：', output_sentence)
        print('')


# 自由翻译，不选择数据集中的句子，查看模型泛化效果
def translation(encoder, decoder, text):
    in_lang = []
    textbox = [s for s in text.split(' ')]
    # print((textbox))
    # print(len(textbox))
    for i in range(len(textbox)):
        in_lang.append(lang_dataset.input_lang.word2index[textbox[i]])
    in_lang.append(Ed_tag)               # 添加句末标记！
    in_lang = torch.LongTensor(in_lang)  # 需要将列表转为张量传入
    # print(in_lang)
    print('待编配的主旋律声频序列为：', text)
    if use_attention:
        output_words, attentions = evaluate(encoder, decoder, in_lang)
    else:
        output_words = evaluate(encoder, decoder, in_lang)
    output_sentence = ' '.join(output_words)
    print('模型自动编配和弦序列为：', output_sentence)
    print()
    return output_sentence


input_size = lang_dataset.input_lang_words
hidden_size = 128
embed_dim = 56
output_size = lang_dataset.output_lang_words
if use_attention:
    decoder = AttentionDecoder_RNN(hidden_size, embed_dim, output_size, 2)
    decoder.load_state_dict(torch.load('./attentiondecoder_param.pkl'))
else:
    decoder = Decoder_RNN(hidden_size, embed_dim,output_size, 1)
    decoder.load_state_dict(torch.load('./decoder_param.pkl'))
encoder = Encoder_RNN(input_size, embed_dim, hidden_size)
encoder.load_state_dict(torch.load('./encoder_param.pkl'))


#evaluateRandomly(encoder, decoder)
#translation(encoder,decoder,"00005353 53535353 53535353 53535353")
# play1=translation(encoder,decoder,"46464646 46464444 53535353 53534444 48484949 48444444 53535353 53535353")
# play2=translation(encoder,decoder,"46464646 46464444 41414141 44444646 48484949 48444444 46464646 46464646")
# play3=translation(encoder,decoder,"46464646 46464444 48484848 48484949 48484949 51494949 41414141 51514949")
# play4=translation(encoder,decoder,"46464646 41414141 48484848 48484444 46464646 46464646 46464646 46464646")

#《走马》的四节旋律，进行翻译
play1=translation(encoder,decoder,"37374646 46464646 46464646 44444444 00003939 39394141 39393939 39393939")
#sugar1=translation(encoder,decoder,"00003939 39394141 39393939 39393939 39393939 39393939 39393939 34343737")
play2=translation(encoder,decoder,"39393939 39393939 39393939 34343737 00003939 39394141 41414141 39393737")
play3=translation(encoder,decoder,"37374646 46464646 46464646 44444444 00003939 39393939 00000000 39393737")
#sugar2=translation(encoder,decoder,"00003939 39393939 00000000 39393737 41414444 44444444 44444444 44444444")
play4=translation(encoder,decoder,"41414444 44444444 44444444 44444444 00000000 00000000 00000000 00000000")


#随机挑出一句查看它的注意力分布
# if use_attention:
#     pair_idx = random.choice(list(range(len(lang_dataset))))
#     pairs = lang_dataset.pairs[pair_idx]
#     print('>')
#     #print(pairs[0])
#     in_lang, out_lang = lang_dataset[pair_idx]
#     output_words, attentions = evaluate(encoder, decoder, in_lang)
#     # 矩阵可视化，画出注意力模型
#     plt.matshow(attentions.cpu().numpy())
#     plt.show()