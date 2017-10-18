# coding=utf-8

from src.utils.vocab_utils import NUM


class MyIOError(Exception):
    """
    文件读取错误
    """
    def __init__(self, filename):
        """
        实例初始化
        参数：
            ·filename    -I  文件名
        返回值：无
        """
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """
    数据处理类
    """
    def __init__(self,
                 filename,
                 lowercase=False,
                 use_char=False,
                 max_chars=50,
                 max_words=None):
        """
        实例初始化
        参数：
            filename            -I  文件名
            processing_words    -I  处理单词函数
            processing_tags     -I  处理标签函数
            max_iter            -I  最多生成的句子数
        返回值：无
        """
        self.filename = filename
        self.lowercase = lowercase
        self.use_char = use_char
        self.max_chars = max_chars
        self.max_words = max_words
        self.length = None

    def __iter__(self):
        """
        迭代器继承
        参数：
            无
        返回值：
            words   -O  句子中单词信息
            tags    -O  句子中标注信息
        """
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    word = processing_word(word,
                                           max_chars=self.max_chars,
                                           lowercase=self.lowercase,
                                           use_char=self.use_char)
                    tag = processing_word(tag,
                                          lowercase=False,
                                          use_char=False)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """
        迭代器继承
        参数：
            无
        返回值：
            words   -O  句子中单词信息
            tags    -O  句子中标注信息
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def processing_word(word, max_chars=None, lowercase=False, use_char=False):
    """
    得到处理单词的函数句柄
    参数：
        vocab_words     -I      单词词典        dict[word] = idx
        vocab_chars     -I      字符词典
        lowercase       -I      是否进行小写转换
        chars           -I      是否进行字符级别处理
    返回值：
        f               -O      函数句柄
        f("cat") = ([12, 4, 32], 12345) = (list of char ids, word id)
    """
    # 0. get chars of words
    if use_char == True:
        chars = []
        for char in word:
            chars += [char]
            if max_chars is not None and len(chars) >= max_chars:
                break

    # 1. preprocess word
    if lowercase:
        word = word.lower()
    if word.isdigit():
        word = NUM

    # 2. return tuple char ids, word id
    if use_char == True:
        return chars, word
    else:
        return word

