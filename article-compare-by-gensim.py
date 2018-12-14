#coding=utf8
'''
使用gensim，行简单的文本相似度分析
'''
import jieba.analyse
from gensim import corpora, models, similarities

jieba.analyse.set_stop_words('./stopwords.txt')
# 以下doc0-doc7是几个最简单的文档，我们可以称之为目标文档，本文就是分析doc_test（测试文档）与以上8个文档的相似度。
doc0 = "我不喜欢上海"
doc1 = "上海是一个好地方"
doc2 = "北京是一个好地方"
doc3 = "上海好吃的在哪里"
doc4 = "上海好玩的在哪里"
doc5 = "上海是好地方"
doc6 = "上海路和上海人"
doc7 = "喜欢小吃"
doc_test="我喜欢上海的小吃"

# 首先，为了简化操作，把目标文档放到一个列表all_doc中。
all_doc = []
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)
all_doc.append(doc5)
all_doc.append(doc6)
all_doc.append(doc7)

# 以下对目标文档进行分词，并且保存在列表all_doc_list中
all_doc_list = []
for doc in all_doc:
    words = jieba.cut(doc)

    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)

# 以下把测试文档也进行分词，并保存在列表doc_test_list中
doc_test_list = [word for word in jieba.cut(doc_test)]


###############################制作语料库############################
# 首先用dictionary方法获取词袋（bag-of-words)
dictionary = corpora.Dictionary(all_doc_list)
# 词袋中用数字对所有词进行了编号
dictionary.keys()
# 以下使用doc2bow制作语料库
corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

# 以下用同样的方法，把测试文档也转换为二元组的向量
doc_test_vec = dictionary.doc2bow(doc_test_list)



############################相似度分析################################
# 使用TF-IDF模型对语料库建模
tfidf = models.TfidfModel(corpus)
# 获取测试文档中，每个词的TF-IDF值

# 对每个目标文档，分析测试文档的相似度
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
sim = index[tfidf[doc_test_vec]]

# 根据相似度排序
result = sorted(enumerate(sim), key=lambda item: -item[1])
print "*" * 100
print result


'''
从分析结果来看，测试文档与doc7相似度最高，其次是doc0，与doc2的相似度为零。大家可以根据TF-IDF的原理，看看是否符合预期。 
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

'''