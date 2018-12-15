#coding=utf8
'''
gemsim是一个免费机器学习的python库，设计目的是，从文档中有效地自动抽取语义主题。
gensim可以处理原始的，非结构化的文本（”plain text”）
具体的gensim的介绍，可以看这个文章，通俗易懂：
https://zhuanlan.zhihu.com/p/37175253

本文做的事情是
1、帮助大家理解一下gansim的基础概念，比较通俗易懂
2、这里的其实做了一个单词的权重打分函数，和结巴的分词打分，差距比较小，大多数在2倍，区间大概在1-4倍的样子,所以我认为还比较准确的
3、这里只是个demo，真正实现这个功能，还需要，动态加载和主动学习，保扩效率等
'''

import jieba.analyse
from gensim import corpora
jieba.analyse.set_stop_words('./src/stopwords.txt')

jieba_word_score = {}

def split_words_fun(content, need_print=False):
    seg = jieba.cut(content)
    # 设置结巴的stopwords， 就是不需要进行处理或者是不重要的 string
    if split_words_mode == 'TF-IDF':
        key_word = jieba.analyse.extract_tags('|'.join(seg), topK=100, withWeight=True, allowPOS=())
    elif split_words_mode == 'TextRank':
        key_word = jieba.analyse.textrank('|'.join(seg), topK=100, withWeight=True, allowPOS=())
    else:
        raise Exception

    _key_words = []
    if need_print:
        print 'jieba 分词整理出来的关键字权重'
    for one in key_word:
        if need_print:
            print one[0], one[1], '   ,',
            jieba_word_score[one[0]] = one[1]
        _key_words.append(one[0])

    return _key_words


def gansim_learning():
    '''
    gansim 入门学习，使用例子，来更好地理解，gansim的使用方法
    Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。
    它支持包括TF-IDF，LSA，LDA，和word2vec在内的多种主题模型算法，
    支持流式训练，并提供了诸如相似度计算，信息检索等一些常用任务的API接口
    参考文档：https://blog.csdn.net/DuinoDu/article/details/76618638
    :return:
    '''
    # 语料如下
    with open(r'./src/article-content.txt', 'r') as file:
        raw_corpus = file.readlines()
    raw_corpus = [article.decode('utf8') for article in raw_corpus]
    print raw_corpus

    # 需要做一些预处理。抽取文档中最具有代表性的关键词
    precessed_corpus = []
    for content in raw_corpus:
        key_word = split_words_fun(content)
        precessed_corpus.append(key_word)

    # 在正式处理之前，我们想对语料中的每一个单词关联一个唯一的ID。
    # 这可以用gensim.corpora.Dictionary来实现。这个字典定义了我们要处理的所有单词表。
    dictionary = corpora.Dictionary(precessed_corpus)
    # 由于我们的语料不大，在我们定义的字典中，只有38个不同的单词。对于更大一些的语料，包含上百万个单词的字典是很常见的。
    print '我们的语料库制作的字典，包含词语和词语在字典中的id'
    id2token = {v: k for k, v in dictionary.token2id.iteritems()}
    print dictionary.token2id

    # 下一步就是要对我们的单词列表进行向量化
    # 一种常见的方法是bag - of - words模型。在词袋模型中，每篇文档表示被表示成一个向量，代表字典中每个词出现的次数。
    # 例如， 给定一个包含 dic = [‘coffee’, ’milk’, ’sugar’, ’spoon’]的字典，一个包含 str = [‘coffee milk coffee’]字符串的文档
    # 遍历 dic ,如果匹配到文中关键字，就给字典对应位置，加1，前文中的str  可以表示成向量[2, 1, 0, 0]。
    # 词袋模型的一个重要特点是，它完全忽略的单词在句子中出现的顺序，这也就是“词袋”这个名字的由来。

    # 事实上，真正的dic，每一个单词有自己的一个id。比如{'coffee': 3, 'milk': 1, 'sugar': 2, 'spoon’: 0}
    # 那么。str其实表示为：[(3, 2), (1,1)], 这个元祖中，第一个元素表示单词的id， 第二个表示单词出现的次数，它的顺序还是跟dic的顺序保持一致的！
    # 那么，如果是没有被dict收录的单词呢，我们的做法是跳过，当然，也可以动态的加进去，这个我还没有弄过，不知道

    # 接着上面的代码，我们可以把原始的语料转化为一组向量：
    # 文中使用的方式是，所有的数据都一股脑的怼到内存中，如果是很多的数据量，可以用迭代的方式，
    # 不然，内存可是会爆炸的喔
    bow_corpus = [dictionary.doc2bow(text) for text in precessed_corpus]
    print '使用语料库和字典，制作的稀疏向量的结果集'
    print bow_corpus

    #######################################################################################################
    # 前期的语料库已经准备结束了,下面就可以做很多好玩的事情了。
    # 比如 1、我们现在使用语料库，制作一个tf-idf,用我们的语料库进行训练，就可以成为一个tf-idf的库了。
    # 2、我们可以用它作为文本比对的库，
    # 这些具体的理论，可以参考官方文档的翻译： https://blog.csdn.net/questionfish/article/details/46725475
    # 我们现在先制作一个tf-idf的库来试试水
    from gensim import models
    tfidf = models.TfidfModel(bow_corpus)
    with open(r'./src/article-content-test.txt', 'r') as file:
        test_string = file.readlines()
        test_string = test_string[0]
    string_bow = dictionary.doc2bow(split_words_fun(test_string, True))
    string_tfidf = tfidf[string_bow]
    print '*' * 100
    print '【待解析文本】的 稀疏向量'
    print string_bow
    print '根据语料库制作的 tf-idf算法，对稀疏向量进行 tf-idf 计算'
    print string_tfidf
    # 要把ft-idf计算的结果，转化成最初的可识别文本
    string_tfidf_result = [(id2token.get(word_tfidf[0]), word_tfidf[1]) for word_tfidf in string_tfidf]
    print '【待解析文本】的 关键字权重 结果'
    for one in string_tfidf_result:
        print one[0], '\tgansim 权重=', one[1], '\t结巴权重=', jieba_word_score.get(one[0]), '\tgansim 权重 / 结巴权重=',one[1]/jieba_word_score.get(one[0])
    return string_tfidf_result

if __name__  == '__main__':
    split_words_mode = 'TF-IDF'
    gansim_learning()