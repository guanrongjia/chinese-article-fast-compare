# -*- coding: utf-8 -*-
'''
海量文章查重脚本，
powered by guanrongjia

海量比对文章的相似度！
这个算法对文字量比较大的文章，查询相似度，还是比较准的，
大于10 的(精度可以根据自己需要，一半设置为3，那是非常精确了)，为不相似，否则为极度相似。

但是对于那种只摘抄了一部分的，相似度查询，很差
如果要根据段落来，甚至是句子来，也可以，但是工作量就比较大了，
也每一句话都存一个simhash。.这样就可以做到按照句子来比对了。

此文本为初始版本，最新版本，使用 test-article-compare2.py
内含多个不同的比对方式，可以直接运行demo 看结果

'''
import jieba
import jieba.analyse
import numpy as np
import re

def calc_simhash(content):
    '''
    对一个文章，计算出他的simhash
    :param content:
    :return:
    '''

    # 结巴分词
    seg = jieba.cut(content)
    # 设置结巴的stopwords， 就是不需要进行处理或者是不重要的 string
    jieba.analyse.set_stop_words('./src/stopwords.txt')
    # 使用 extract 算法，计算出关键词和 关键词得分（权重）
    keyWord = jieba.analyse.extract_tags('|'.join(seg), topK=25, withWeight=True, allowPOS=())
    keyList = []
    for feature, weight in keyWord:
        weight = int(weight * 100)
        # 关键词hash化， 转化成 1和0
        feature = string_hash(feature)
        temp = []
        for i in feature:
            if(i == '1'):
                temp.append(weight)
            else:
                temp.append(-weight)
        # 计算每一个关键词的最终得分
        keyList.append(temp)
    print ''
    # 计算整个文章中，找出的关键词（文中设置的是100），的得分相加
    list1 = np.sum(np.array(keyList), axis=0)

    if(keyList==[]): #编码读不出来
        return '00'
    simhash = ''
    for i in list1:
        if(i > 0):
            simhash = simhash + '1'
        else:
            simhash = simhash + '0'
    return simhash


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
    return str(x)

def hamming_dis(simhash1, simhash2):
    '''
    使用位运算，比较两个文章最终 关键hash
    :param simhash1:
    :param simhash2:
    :return:
    '''
    t1 = '0b' + simhash1
    t2 = '0b' + simhash2
    n=int(t1, 2) ^ int(t2, 2)
    i=0
    while n:
        n &= (n-1)
        i+=1
    return i


if __name__ == '__main__':
    simhash_1 = calc_simhash(u'''我是独生女，他们的希望都在我这里 。越是这样想的越多，想的越多越痛苦，我快要发疯啦，我的性格本身就急躁，再加上这件事压抑着，谁都无法倾诉，更加抑郁。
　　今天，我去相亲，他看见了，我在这一年里总是挣扎着闹着说分手，但总是狠不下心，觉得他想我身体的一部分一样珍贵而难以割舍，爱他已经成为我的习惯。每次因为他没有办法陪在我身边而生气，我都很懊恼，我决心一定要离开他，发誓一定要找到一个能陪我的人，但总是遇不到。每次我闹，他都耐心哄我，我们就又在一起了，我痛恨自己对他的依赖，我自认是个自立的女孩，但这是我第一次如此深切的爱一个，我做不到。
　　今天，我说，哥哥，请你放开我的手吧，我求他给我希望，哪怕是骗我，我都会等他的，但他从来没说过，他说不能拿我的幸福开玩笑。可是，他应该知道，我的幸福就是他啊，但现实让我们不得不以地下的关系维持着这份爱。也许好多人会鄙视我，但爱是真的，就该得到尊重。今天，我在他怀里哭了，他说叫我把握好能给我幸福的人，以后有什么委屈哥哥的肩膀永远向你敞开。临分开时，我对他说，哥哥，我爱你，他说我永远爱你，我想，我们正式结束了，我回到家里大哭了一场。
　　今天，我说，哥哥，请你放开我的手吧，我求他给我希望，哪怕是骗我，我都会等他的，可是，他应该知道，我的幸福就是他啊，但现实让我们不得不以地下的关系维持着这份爱。也许好多人会鄙视我，但爱是真的，就该得到尊重。
    。今天，我在他怀里哭了，他说叫我把握好能给我幸福的人，以后有什么委屈哥哥的肩膀永远向你敞开。临分开时，我对他说，哥哥，我爱你，他说我永远爱你，我想，我们正式结束了，我回到家里大哭了一场。''')

    simhash_2 = calc_simhash(u'''我是独生女，他们的希望都在我这里 。越是这样想的越多，想的越多越痛苦，我快要发疯啦，我的性格本身就急躁，再加上这件事压抑着，谁都无法倾诉，更加抑郁。
　　今天，我去相亲，他看见了，我在这一年里总是挣扎着闹着说分手，但总是狠不下心，觉得他想我身体的一部分一样珍贵而难以割舍，爱他已经成为我的习惯。每次因为他没有办法陪在我身边而生气，我都很懊恼，我决心一定要离开他，发誓一定要找到一个能陪我的人，但总是遇不到。每次我闹，他都耐心哄我，我们就又在一起了，我痛恨自己对他的依赖，我自认是个自立的女孩，但这是我第一次如此深切的爱一个，我做不到。
　　今天，我说，哥哥，请你放开我的手吧，我求他给我希望，哪怕是骗我，我都会等他的，但他从来没说过，他说不能拿我的幸福开玩笑。可是，他应该知道，我的幸福就是他啊，但现实让我们不得不以地下的关系维持着这份爱。也许好多人会鄙视我，但爱是真的，就该得到尊重。今天，我在他怀里哭了，他说叫我把握好能给我幸福的人，以后有什么委屈哥哥的肩膀永远向你敞开。临分开时，我对他说，哥哥，我爱你，他说我永远爱你，我想，我们正式结束了，我回到家里大哭了一场。
　　今天，我说，哥哥，请你放开我的手吧，我求他给我希望，哪怕是骗我，我都会等他的，''')

    # 计算最终差异度
    diff_value =  hamming_dis(simhash_1, simhash_2)
    print u'差异值：  %s' % (diff_value)
    # if diff_value > 8:
    #     print u'差异值：  %s' % (diff_value)