'''
处理数据集的相关代码，目前我的数据集模式是仿照NMT语言翻译来的，但具体每一个样本的选取、结构是根据声乐知识制作完成的
对每一组旋律-和弦样本进行处理装入dataloader供接下来的训练使用

silence   2019.12.14
'''

import random
import re
import unicodedata
import torch
from torch.utils.data import Dataset

St_tag=0   #句子起始标志
Ed_tag=1   #句子结束标志
MAXLEN=15


class language2vec(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}                      #单词查找序号
        self.word2count = {}                      #单词在数据集中出现的次数
        self.index2word = {0: "St", 1: "Ed"}      #序号查找单词
        self.n_words = 2                          #字典总量
    #处理数据分出单词
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    #处理单词
    def addWord(self, word):
        #未出现过便新增在字典中
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

#将unicode转为ascii，并化为小写去除标点(音乐文本不需要，和弦我们使用常规英文字母表示)
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn')
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s

#读入所有数据并处理
def readLanguage(l1, l2, reverse=False):
    lines = open('./dataset/%s-%s.txt' % (l1, l2),encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('@')] for l in lines]
    #print(pairs)
    #两方由谁译谁
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = language2vec(l2)
        output_lang = language2vec(l1)
    else:
        input_lang = language2vec(l1)
        output_lang = language2vec(l2)
    return input_lang, output_lang, pairs


#为简化计算过滤掉不符条件的数据集（之前做英-法互译时用的，因为数据集太大训练不过来，加一个过滤缩小数据集，音乐自制数据集本来就少，不再需要了）
eng_prefixes = ("")      #空则为不做过滤处理
def filterPair(p):
    return len(p[0].split(' ')) <= MAXLEN and len(p[1].split(' ')) <= MAXLEN and p[0].startswith(eng_prefixes)
def getPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


#加载数据，处理翻译对训练集，返回序列化的训练集（待译句子文本和目标句子文本）
def prepareData(l1, l2, reverse=False):
    #读入所有数据处理为vec序列
    input_lang, output_lang, pairs = readLanguage(l1, l2, reverse)
    #按条件过滤
    pairs = getPairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    #计算过滤后出现的单词
    print("单词计数:")
    print("待译语言(旋律)字典单词数目：",'%d' %input_lang.n_words)
    print("翻译语言(和弦)字典单词数目：",'%d' %output_lang.n_words)
    #print(input_lang.index2word)
    #print(output_lang.index2word)
    return input_lang, output_lang, pairs


#句子转张量
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(Ed_tag)
    result = torch.LongTensor(indexes)
    return result
#句子翻译对转张量
def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor


#根据上面的函数定义自己的数据集类，重写两个函数
class MyDataset(Dataset):
    def __init__(self, dataload=prepareData, lang=['和弦3', '旋律3']):
        self.input_lang, self.output_lang, self.pairs = dataload(lang[0], lang[1],True)
        #数量
        self.input_lang_words = self.input_lang.n_words
        self.output_lang_words = self.output_lang.n_words
    def __getitem__(self, index):
        #根据序号返回这一对翻译句子的数字化张量
        return tensorFromPair(self.input_lang, self.output_lang,self.pairs[index])
    def __len__(self):
        return len(self.pairs)

#test=MyDataset()