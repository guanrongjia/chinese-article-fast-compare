#coding=utf8
'''
使用gensim，行简单的文本相似度分析


本文做的事情是
1、帮助大家理解一下gansim的基础概念，比较通俗易懂
2、这里的其实做了简单的文章比对工具，需要把比对的文件也加入语料库，这样数据比较准确
3、这里只是个demo，真正实现这个功能，还需要，动态加载和主动学习，保扩效率等
4、如果需要看其他的文本比对方法，或者需要下载完整的demo以及资源文件， 请关注我的git项目：
https://github.com/guanrongjia/chinese-article-fast-compare


具体的gensim的介绍，可以看这个文章，通俗易懂：
https://zhuanlan.zhihu.com/p/37175253
本文参考文档：
https://blog.csdn.net/xiexf189/article/details/79092629

'''
import jieba.analyse
from gensim import corpora, models, similarities
jieba.analyse.set_stop_words('./src/stopwords.txt')


def split_words_fun(content, need_print=False):
    '''
    功能函数，根据结巴分词，切分长句子
    :param content:
    :param need_print:
    :return:
    '''
    seg = jieba.cut(content)
    # 设置结巴的stopwords， 就是不需要进行处理或者是不重要的 string
    if split_words_mode == 'TF-IDF':
        key_word = jieba.analyse.extract_tags('|'.join(seg), topK=100, withWeight=True, allowPOS=())
    elif split_words_mode == 'TextRank':
        key_word = jieba.analyse.textrank('|'.join(seg), topK=100, withWeight=True, allowPOS=())
    else:
        raise Exception

    return [one[0] for one in key_word]

def gansim_demo():
    '''
    下面是一个具体的，使用gansim进行文本比对，相似度计算的例子
    :return:
    '''
    ###############################预处理语料############################
    #  把训练材料准备好
    with open(r'./src/article-content.txt', 'r') as file:
        raw_corpus = file.readlines()
    # 把我们的比对材料也加入语料库
    all_doc = ["我不喜欢上海",
               "上海是一个好地方",
               "北京是一个好地方",
               "上海好吃的在哪里",
               "上海好玩的在哪里",
               "上海是好地方",
               "上海路和上海人",
               "喜欢小吃"]
    raw_corpus.extend(all_doc)
    raw_corpus = [article.decode('utf8') for article in raw_corpus]
    # 需要做一些预处理。抽取文档中最具有代表性的关键词
    precessed_corpus = []
    for content in raw_corpus:
        key_word = split_words_fun(content)
        precessed_corpus.append(key_word)

    ###############################制作语料库############################
    # 首先用dictionary方法获取词袋
    dictionary = corpora.Dictionary(precessed_corpus)
    # 以下使用doc2bow制作语料库
    bow_corpus = [dictionary.doc2bow(doc) for doc in precessed_corpus]
    # 使用TF-IDF模型对语料库建模
    tfidf = models.TfidfModel(bow_corpus)

    ############################处理待检测数据###########################
    doc_test = "我喜欢上海的小吃"
    # 以下对目标文档进行分词，并且保存在列表all_doc_list中
    all_doc_list = []
    for doc in all_doc:
        words = jieba.cut(doc)
        doc_list = [word for word in words]
        all_doc_list.append(doc_list)
    all_corpus = [dictionary.doc2bow(text) for text in all_doc_list]

    # 以下把测试文档也进行分词，并保存在列表doc_test_list中
    doc_test_list = [word for word in jieba.cut(doc_test)]
    test_corpus = dictionary.doc2bow(doc_test_list)

    ############################相似度分析################################

    # 对每个目标文档，分析测试文档的相似度
    index = similarities.SparseMatrixSimilarity(tfidf[all_corpus], num_features=len(dictionary.token2id))
    sim = index[tfidf[test_corpus]]

    # 根据相似度排序
    result = sorted(enumerate(sim), key=lambda item: -item[1])
    print "*" * 100
    print result


    '''
    最终结果：
    [(7, 0.86156476), (0, 0.5689938), (6, 0.50764775), (1, 0.4041334), (5, 0.4041334), (3, 0.34782335), (4, 0.31870633), (2, 0.0)]
    从分析结果来看，测试文档与doc7相似度最高，其次是doc0,doc6.
    与doc2的相似度为零。
    大家可以根据TF-IDF的原理，看看是否符合预期。 
    最后总结一下文本相似度分析的步骤：
    
    读取文档
    对要计算的多篇文档进行分词
    对文档进行整理成指定格式，方便后续进行计算
    计算出词语的词频
    【可选】对词频低的词语进行过滤
    建立语料库词典
    加载要对比的文档
    将要对比的文档通过doc2bow转化为词袋模型
    对词袋模型进行进一步处理，得到新语料库
    将新语料库通过tfidfmodel进行处理，得到tfidf
    通过token2id得到特征数
    稀疏矩阵相似度，从而建立索引 
    得到最终相似度结果
    
    参考文档：https://blog.csdn.net/xiexf189/article/details/79092629
    '''



if __name__  == '__main__':
    split_words_mode = 'TF-IDF'
    gansim_demo()