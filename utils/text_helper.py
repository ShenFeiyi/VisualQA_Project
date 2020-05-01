# 用于将句子单词转换为相应的索引列表等
import re


SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # 用于切词，标点符号会被保留下来


def tokenize(sentence):
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [ t.strip() for t in tokens if len(t.strip()) > 0 ]
    return tokens # 返回句子切词列表


def load_str_list(fname): # 用户载入词典文件，转换为列表
    with open(fname) as f:
        lines = f.readlines() # 返回列表
    lines = [ l.strip() for l in lines ] # 一个字典一个列表一个str
    return lines


class VocabDict:

    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = { w:n_w for n_w, w in enumerate(self.word_list) } # 单词索引字典
        self.vocab_size = len(self.word_list)
        self.unk2idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None # 未知词索引

    def idx2word(self, n_w):

        return self.word_list[n_w] # 返回索引为n_w的单词

    def word2idx(self, w): # 返回单词在字典中的索引
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None: # 字典中存在<unk>,返回其索引
            return self.unk2idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [ self.word2idx(w) for w in tokenize(sentence) ]

        return inds # 将句子中的单词转换为索引列表

if __name__=="__main__":
    t = tokenize('How are you* today?') # ['how', ' ', 'are', ' ', 'you', '* ', 'today', '?']

