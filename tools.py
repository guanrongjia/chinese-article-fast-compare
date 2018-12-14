#coding=utf8
import jieba
import jieba.posseg as pseg


def splitWords(str_a):
    '''
    接受一个字符串作为参数，返回分词后的结果字符串(空格隔开)和集合类型
    '''
    wordsa = pseg.cut(str_a)
    cuta = ""
    seta = set()
    for key in wordsa:
        cuta += key.word + " "
        seta.add(key.word)
    return [cuta, seta]



def countIDF(text, topK, split_words_mode):
    '''
    text:字符串，topK根据TF-IDF得到前topk个关键词的词频，用于计算相似度
    return 词频vector
    '''
    # 结巴分词
    seg = pseg.cut(text)
    # 设置结巴的stopwords， 就是不需要进行处理或者是不重要的 string
    jieba.analyse.set_stop_words('./stopwords.txt')
    # 使用 extract 算法，计算出关键词和 关键词得分（权重）
    if split_words_mode == 'TF-IDF':
        keyWord = jieba.analyse.extract_tags('|'.join(seg), topK=topK, withWeight=True, allowPOS=())
    elif split_words_mode == 'TextRank':
        keyWord = jieba.analyse.textrank('|'.join(seg), topK=topK, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    else:
        raise Exception
    return [int(one[1] * 20) for one in keyWord]



def string_hash(source):  # 局部哈希算法的实现
    if source == "":
        return 0
    else:
        # ord()函数 return 字符的Unicode数值
        x = ord(source[0]) << 7
        m = 1000003  # 设置一个大的素数
        mask = 2 ** 128 - 1  # key值
        for c in source:  # 对每一个字符基于前面计算hash
            x = ((x * m) ^ ord(c)) & mask

        x ^= len(source)  #
        if x == -1:  # 证明超过精度
            x = -2
        x = bin(x).replace('0b', '').zfill(64)[-64:]
        # print(source,x)

    return str(x)