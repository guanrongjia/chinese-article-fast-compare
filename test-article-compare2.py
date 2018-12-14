# coding: utf-8
'''
各个 文本查重算法以及demo比对
文本相似度的计算，基于几种常见的算法的实现
'''

import jieba
from jieba import analyse
import numpy as np
from tools import splitWords, countIDF, string_hash


class TextSimilarity(object):

    def __init__(self, str_a, str_b):
        '''
        初始化类行
        '''
        self.str_a = str_a
        self.str_b = str_b

    # get LCS(longest common subsquence),DP
    def lcs(self, str_a, str_b):
        lensum = float(len(str_a) + len(str_b))
        # 得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
        lengths = [[0 for j in range(len(str_b) + 1)] for i in range(len(str_a) + 1)]
        # enumerate(a)函数： 得到下标i和a[i]
        for i, x in enumerate(str_a):
            for j, y in enumerate(str_b):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # 到这里已经得到最长的子序列的长度，下面从这个矩阵中就是得到最长子序列
        result = ""
        x, y = len(str_a), len(str_b)
        while x != 0 and y != 0:
            # 证明最后一个字符肯定没有用到
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            else:  # 用到的从后向前的当前一个字符
                assert str_a[x - 1] == str_b[y - 1]  # 后面语句为真，类似于if(a[x-1]==b[y-1]),执行后条件下的语句
                result = str_a[x - 1] + result  # 注意这一句，这是一个从后向前的过程
                x -= 1
                y -= 1

        longestdist = lengths[len(str_a)][len(str_b)]
        ratio = longestdist / min(len(str_a), len(str_b))
        # return {'longestdistance':longestdist, 'ratio':ratio, 'result':result}
        return ratio

    def minimumEditDistance(self, str_a, str_b):
        '''
        最小编辑距离，只有三种操作方式 替换、插入、删除
        '''
        lensum = float(len(str_a) + len(str_b))
        if len(str_a) > len(str_b):  # 得到最短长度的字符串
            str_a, str_b = str_b, str_a
        distances = range(len(str_a) + 1)  # 设置默认值
        for index2, char2 in enumerate(str_b):  # str_b > str_a
            newDistances = [index2 + 1]  # 设置新的距离，用来标记
            for index1, char1 in enumerate(str_a):
                if char1 == char2:  # 如果相等，证明在下标index1出不用进行操作变换，最小距离跟前一个保持不变，
                    newDistances.append(distances[index1])
                else:  # 得到最小的变化数，
                    newDistances.append(1 + min((distances[index1],  # 删除
                                                 distances[index1 + 1],  # 插入
                                                 newDistances[-1])))  # 变换
            distances = newDistances  # 更新最小编辑距离

        mindist = distances[-1]
        ratio = (lensum - mindist) / lensum
        # return {'distance':mindist, 'ratio':ratio}
        return ratio

    def levenshteinDistance(self, str1, str2):
        '''
        编辑距离——莱文斯坦距离,计算文本的相似度
        '''
        m = len(str1)
        n = len(str2)
        lensum = float(m + n)
        d = []
        for i in range(m + 1):
            d.append([i])
        del d[0][0]
        for j in range(n + 1):
            d[0].append(j)
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:
                    d[i].insert(j, d[i - 1][j - 1])
                else:
                    minimum = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 2)
                    d[i].insert(j, minimum)
        ldist = d[-1][-1]
        ratio = (lensum - ldist) / lensum
        # return {'distance':ldist, 'ratio':ratio}
        return ratio

    def JaccardSim(self, str_a, str_b):
        '''
        Jaccard相似性系数
        计算sa和sb的相似度 len（sa & sb）/ len（sa | sb）
        '''
        seta = splitWords(str_a)[1]
        setb = splitWords(str_b)[1]

        sa_sb = 1.0 * len(seta & setb) / len(seta | setb)

        return sa_sb

    def splitWordSimlaryty(self, str_a, str_b, split_words_mode, topK=25):
        '''
        基于分词求相似度，默认使用cos_sim 余弦相似度,默认使用前20个最频繁词项进行计算
        '''
        # 得到前topK个最频繁词项的字频向量
        def cos_sim(a, b):
            # 文本的余弦相似度
            a = np.array(a)
            b = np.array(b)

            return np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

        def eucl_sim(a, b):
            # 文本的欧几里德相似度
            a = np.array(a)
            b = np.array(b)

            return 1 / (1 + np.sqrt((np.sum(a - b) ** 2)))

        def pers_sim(a, b):
            # 文本的皮尔森相似度
            a = np.array(a)
            b = np.array(b)

            a = a - np.average(a)
            b = b - np.average(b)

            return np.sum(a * b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))
        vec_a = countIDF(str_a, topK, split_words_mode)
        vec_b = countIDF(str_b, topK, split_words_mode)

        return cos_sim(vec_a, vec_b), pers_sim(vec_a, vec_b), pers_sim(vec_a, vec_b)


    def simhash(self, str_a, str_b, split_words_mode):
        '''
        使用simhash计算相似度
        '''
        t1 = '0b' + self.calc_simhash(str_a, split_words_mode)
        t2 = '0b' + self.calc_simhash(str_b, split_words_mode)
        n = int(t1, 2) ^ int(t2, 2)
        i = 0
        while n:
            n &= (n - 1)
            i += 1
        return i

    def calc_simhash(self,content, split_words_mode):
        # 结巴分词
        seg = jieba.cut(content)
        # 设置结巴的stopwords， 就是不需要进行处理或者是不重要的 string
        jieba.analyse.set_stop_words('./stopwords.txt')
        # 使用  TF-IDF / TextRank  算法，           计算出关键词和 关键词得分（权重）
        if split_words_mode == 'TF-IDF':
            keyWord = jieba.analyse.extract_tags('|'.join(seg), topK=25, withWeight=True, allowPOS=())
        elif split_words_mode == 'TextRank':
            keyWord = jieba.analyse.textrank('|'.join(seg), topK=25, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
        else:
            raise Exception
        keyList = []
        for feature, weight in keyWord:
            weight = int(weight * 20)
            # 关键词hash化， 转化成 1和0
            feature = string_hash(feature)
            temp = []
            for i in feature:
                if (i == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            # 计算每一个关键词的最终得分
            keyList.append(temp)

        # 计算整个文章中，找出的关键词（文中设置的是100），的得分相加
        list1 = np.sum(np.array(keyList), axis=0)

        if (keyList == []):  # 编码读不出来
            return '00'
        simhash = ''
        for i in list1:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'
        return simhash


text1 = u'''
假如我有一双翅膀,我会飞到世界各处好玩的地方。 假如我有一双翅膀，我会飞到北京,北京是我国首都，在那里夏天的太阳扰如一个红红的大火球。冬天，雪花飘飘，像几个小仙子从天而降。我们一家穿着厚衣服，戴着小手套，踩在雪白雪自的地上，堆着雪人，打着雪仗，“嘻嘻哈哈”的笑声让我们给雪人也加上了笑脸。雪人和我们一起笑着。 假如我有一双翅膀，我会飞到我妈妈的家乡——海南。那里四季如春。我们来到无边无际的大海，蓝色的天空和大海连成水天一线。风吹过海面，使人神清气的爽。 假如我有一双翅膀，我会飞到美丽的小兴安岭。那一年四季景色诱人是一座巨大的宝库。春天树木抽出新的枝条，长出了嫩绿的叶子。夏天，树木长了得葱葱笼笼，密密层层的枝叶遮住了蓝蓝的天空。 冬天，雪花在天空中飞舞。树上积满白雪，地上盖起了一张白色的被子，美极了！ 如果我有一双翅膀，我会飞遍世界各处美的地方！
'''

text2 = u'''
假如我有一双翅膀,我要去泰兴。冬天，雪花飘飘我们都是一个家，名字叫中国，我们一家穿着厚衣服，戴着小手套，踩在雪白雪自的地上，堆着雪人，打着雪仗，莱文斯坦距离,计算文本的相似度,“嘻嘻哈哈”的笑声让我们给雪人也加上了笑脸。雪人和我们一起笑着。 假如我有一双翅膀，我会飞到我妈妈的家乡——海南。那里四季如春。我们来到无边无际的大海，蓝色的天空和大海连成水天一线。风吹过海面，使人神清气的爽。 假如我有一双翅膀，我会飞到美丽的小兴安岭。那一年四季景色诱人是一座巨大的宝库。春天树木抽出新的枝条，长出了嫩绿的叶子。夏天，树木长了得葱葱笼笼，密密层层的枝叶遮住了蓝蓝的天空。秋天，风吹来落叶在林间翩翩起舞。冬天，雪花在天空中飞舞。树上积满白雪，地上盖起了一张白色的被子，美极了！  ！
'''

text3 = u'''
今天，我在家的后院玩，突然在墙缝里发现了一位“不速之客”，冬天，雪花飘飘我们都是一个家，名字叫中国，我们一家穿着厚衣服，戴着小手套，踩在雪白雪自的地上，堆着雪人，打着雪仗，可以说不请自到。我一下子就抓住“不速之客”——是只特大的蚂蚁，嘿嘿嘿！我的坏劲儿上来了。 我用一小块石头压住它，它很轻松就翻过身来。不好，这位“大佬”要逃了，我随手一拍就把它抓到了，这次我改变方法，用细土撒在它身上，把它的整个身子都埋了起来。但是，它好像一个大力士一样，不断挣扎地从土里钻了出来。咦！怎么找不着这只蚂蚁了呢？这只“大佬”正穿着吉利服开溜中呐！只见它身披土黄色的吉利服，每只小腿都在不停的运转，岂有此理，竟敢在我的眼皮底下逃跑。看本大侠不好好修理你一顿，别怪我心狠手辣。我又将它埋了起来，比上一次要严实了许多，简直就是蚂蚁界的珠穆朗玛峰了！它拼命挣扎，把头和一支胳膊漏了出来，这让我想到了孙悟空被压在五指山的情形，不禁自喜。 不一会有一只和它体型差不多的蚂蚁来了，它们对了下触角，那只蚂蚁就走了，不一会儿，一只庞大的“军队”浩浩汤汤来了。看来这只压着的蚂蚁是个“当官”的吧！哼哼！我还斗不过你们这些虾兵蟹将，只见这些蚂蚁像患上了“冲锋症”一样，不停的奋勇向前，只见我的珠穆朗玛峰被拆得千疮百孔，很快那只蚂蚁获救了。 我的敬意油然而生，一只只小小的蚂蚁，也能有如此强大的力量，可见团结就是力量。
'''
sim_obj = TextSimilarity(text1, text2)

print u'用最小编辑距离，求文本相似度'
print sim_obj.minimumEditDistance(sim_obj.str_a, sim_obj.str_b)

print u'编辑距离——莱文斯坦距离,计算文本的相似度'
print sim_obj.levenshteinDistance(sim_obj.str_a, sim_obj.str_b)

print u'Jaccard相似性系数，用于比较有限样本集之间的相似性与差异性。Jaccard系数值越大，样本相似度越高。'
print sim_obj.JaccardSim(sim_obj.str_a, sim_obj.str_b)

print u'余弦相似度,  欧几里德相似度, 皮尔森相似度 (TextRank分词)'
print sim_obj.splitWordSimlaryty(sim_obj.str_a, sim_obj.str_b, 'TextRank')

print u'余弦相似度,  欧几里德相似度, 皮尔森相似度 (TF-IDF分词)'
print sim_obj.splitWordSimlaryty(sim_obj.str_a, sim_obj.str_b, 'TF-IDF')

print u'sim hash计算文本相似度 (TextRank分词)'
print sim_obj.simhash(sim_obj.str_a, sim_obj.str_b, 'TextRank')

print u'sim hash计算文本相似度 (TF-IDF分词)'
print sim_obj.simhash(sim_obj.str_a, sim_obj.str_b, 'TF-IDF')