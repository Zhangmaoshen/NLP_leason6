#!/usr/bin/env python 
# -*- coding:utf-8 -*-
############################
#Search Tree -> Similar Words
################################
'''
txt_path= 'D:/doct/leason/NLP/leason6/text.txt'
database=open(txt_path,  'r', encoding='UTF-8')
data=database.read()
database.close()
news_content = data.strip(',').split('\\n')
#print(data_to_list[0])


import pandas as pd
content = pd.read_csv(csv_path, encoding='gb18030')
print(content)
content = content.fillna('')#fillna函数填充csv中NaN表示的缺省值
'''
#####################################################################
#news_content = content.tolist()#矩阵变为列表
#######################################################################
'''
import jieba
def cut(string):
    return ' '.join(jieba.cut(string))
#print(cut('这是一个测试'))#cut后返回str类型数据，可再.split( )一下，变成list类型

import re
def token(string):
    return re.findall(r'[\d|\w]+', string)#字符，数字
#print(token(news_content[0]))


########下面是把news_content各元素整理
news_content = [token(n) for n in news_content]
news_content = [' '.join(n) for n in news_content]
news_content = [cut(n) for n in news_content]
news_content = [n.strip(' ') for n in news_content]
#print(news_content[:2])
###########################
####执行完之后文件已经被写进了内容，后续调试不需要重复执行，可以注释掉,
# 否则每次重新更新都会重新把文件编码格式改成非UTF-8

with open('D:/doct/leason/NLP/leason6/news-sentences-cut.txt', 'w') as f:
    for n in news_content:
        f.write(n + '\n')
    #news_content内容写进news-sentences-cut.txt文件
'''
###############################
'''
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
news_word2ve= Word2Vec(LineSentence('D:/doct/leason/NLP/leason6/news-sentences-cut.txt'), size=35, workers=8)
#下面语句会把训练的模型保存一下
news_word2ve.save('D:/doct/leason/NLP/leason6/./MyModel')
news_word2ve.wv.save_word2vec_format('D:/doct/leason/NLP/leason6/./mymodel.txt', binary=False)
#size=35	训练后词向量的维度; workers=8	设置线程数，越高训练速度（前提是你有这么多）
#print(news_word2ve.most_similar('葡萄牙', topn=20))
'''
##########################################上面的训练结果保存后可以全部注释掉，后面就不需要再重复训练了

#############################################################################################
'''
# 下面是遍历搜索寻找近义词的函数get_related_words(),可在上面和下面部分全部注释掉的情况下单独执行
from collections import defaultdict
from gensim.models import Word2Vec
news_word2ve_model = Word2Vec.load('D:/doct/leason/NLP/leason6/./MyModel')#加载保存的训练后的模型

def get_related_words(initial_words, model):
    unseen = initial_words
    seen = defaultdict(int)
    max_size = 500  # could be greater

    #dict =defaultdict( factory_function):
    # factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，
    # 比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0

    while unseen and len(seen) < max_size:#max_size定义了搜索的次数，应该是越高越准确
        if len(seen) % 50 == 0:
            print('seen length : {}'.format(len(seen)))
        node = unseen.pop(0)#删除key对应的value并返回删除值
        new_expanding = [w for w, s in model.most_similar(node, topn=20)]#遍历查找词节点相似的排名前20的词
        unseen += new_expanding
        seen[node] += 1
        #seen[node] += 1：这样可以统计重复出现的词频，dict（seen）的value值就是词频
        # optimal: 1. score function could be revised
        # optimal: 2. using dymanic programming to reduce computing time
    return seen

print(type(news_word2ve_model))
print(type(news_word2ve_model.wv.vocab))#我理解是把news_word2ve_model变成dict格式，key是词，value是词向量
print(len(news_word2ve_model.wv.vocab))
related_words = get_related_words(['说', '表示'], news_word2ve_model)
#news_word2ve_model是gensim中word2vec的模型的默认格式
#print(related_words)
print(sorted(related_words.items(), key=lambda x: x[1], reverse=True))
#sorted(related_words.items(), key=lambda x: x[1], reverse=True)是按照seen中value排序的，词频高排在前面
#.items()以列表返回可遍历的(键, 值) 元组数组
'''
####################################################################################

########################################################################################

txt_path= 'D:/doct/leason/NLP/leason6/news-sentences-cut.txt'
news_content=open(txt_path,  'r', encoding='UTF-8')
news_content = [n.strip(' ') for n in news_content]
#每个元素是一条新闻
#print(type(news_content))
#print(news_content[11])
#print(len(news_content))
#n.strip(char)除去字符串n中指定字符char
# 同时把news_content分成多个不同元素
'''
print(type(news_content))#这里要注意，没懂这时news_content的类型是啥
print(len(news_content))
print(news_content)
'''


def document_frequency(word):#统计news_content中的word出现的频次
    return sum(1 for n in news_content if word in n)
#print(document_frequency('的'))

import math
def idf(word):#10为底的对数（新闻条数/某个word的出现频次）
    """Gets the inversed document frequency"""
    return math.log10(len(news_content) / document_frequency(word))
#print(idf('的'))
#idf('的') < idf('小米')

def tf(word, document):#统计word在某条新闻中出现的频次
    """
    Gets the term frequemcy of a @word in a @document.
    """
    words = document.split()
    return sum(1 for w in words if w == word)
#print(news_content[11])
#print(tf('垃圾', news_content[11]))

def get_keywords_of_a_ducment(document):#根据词的(tf*idff)数据得到关键词
    words = set(document.split())
    #tfidf=[(w , w在某条新闻-document中出现的频次)]
    tfidf = [
        (w, tf(w, document) * idf(w)) for w in words
    ]
    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)#按（tf*idf)高到低排序
    return tfidf
#print(news_content[0])
#print(get_keywords_of_a_ducment(news_content[0]))
#print(len(news_content))
#print(get_keywords_of_a_ducment(news_content[1500]))

#############################################################Wordcloud
#关于词云，Wordcloud
'''
#print(news_content[500])
#print(get_keywords_of_a_ducment(news_content[500]))
machine_new_keywords = get_keywords_of_a_ducment(news_content[500])#[(词,tf*idf), ()]
machine_new_keywords_dict = {w: score for w, score in machine_new_keywords}#字典

import matplotlib.pyplot as plt
import wordcloud

wc = wordcloud.WordCloud('D:/doct/leason/NLP/leason6/SourceHanSerifSC-Regular.otf')#载入模板文件

#wc.generate_from_frequencies(machine_new_keywords_dict):输入数据，限定显示模式
plt.imshow(wc.generate_from_frequencies(machine_new_keywords_dict))
plt.show()
'''
##################################################TFIDF Vectorized-TF-IDF向量化

sample_num = 1000
sub_samples = news_content[:sample_num]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorized = TfidfVectorizer(max_features=10000)
X = vectorized.fit_transform(sub_samples)
#print(X.shape)#.shape返回数组维度，（r行， c列）
#print(X)
'''
#X.shape
print(X)
print(type(X))#<class 'scipy.sparse.csr.csr_matrix'>
'''

import random
document_id_1, document_id_2 = random.randint(0, 1000), random.randint(0, 1000)
random_choose = random.randint(0, 1000)
'''
news_content[document_id_1]
news_content[document_id_2]
'''
X_id1=vector_of_d_1 = X[document_id_1].toarray()[0]
X_id2=vector_of_d_2 = X[document_id_2].toarray()[0]
#print(X_id1)
#print(X_id2)

from scipy.spatial.distance import cosine
def distance(v1, v2): return cosine(v1, v2)#用cos来定义向量距离
'''
print(distance([1, 1], [2, 2]))
print(distance(X[random_choose].toarray()[0], X[document_id_1].toarray()[0]))
print(distance(X[random_choose].toarray()[0], X[document_id_2].toarray()[0]))
'''
###################################################################################
'''
Build Search Engine
Input: Words
Output: Documents
'''
def naive_search(keywords):#在news_content中匹配keywords
    news_ids = [i for i, n in enumerate(news_content) if all(w in n for w in keywords)]
    return news_ids
#print(naive_search('垃圾 回收'.split()))

import numpy as np
#print(np.where(transposed_x[3000])[0])#没懂？？？
#print(transposed_x[3000])
#1. np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y
# np.where(condition)
# 输出满足条件 (即非0) 元素的坐标

word_2_id = vectorized.vocabulary_#统计词频？？？
id_2_word = {d: w for w, d in word_2_id.items()}#.items()遍历dict的key和value
#print(id_2_word[2000])
#print(word_2_id['今天'])
transposed_x = X.transpose().toarray()#.transpose()对X矩阵求逆


#d1, d2, d3 = {1, 2, 3}, {4, 5, 6, 3, 2}, {1, 3, 4}
#print(reduce(and_, [d1, d2, d3]))#and_：按位与之后的结果
from functools import reduce
from operator import and_

"""
@query is the searched words, splited by space
@return is the related documents which ranked by tf-idf similarity
"""
def search_engine(query):
    words = query.split()
    query_vec = vectorized.transform([' '.join(words)]).toarray()[0]
    candidates_ids = [word_2_id[w] for w in words]
    documents_ids = [
        set(np.where(transposed_x[_id])[0]) for _id in candidates_ids#没懂
    ]
    merged_documents = reduce(and_, documents_ids)
    # we could know the documents which contain these words
    sorted_docuemtns_id = sorted(merged_documents, key=lambda i: distance(query_vec, X[i].toarray()))
    return sorted_docuemtns_id
#print(np.where(vectorized.transform(['垃圾 回收']).toarray()[0]))



text = """新华社洛杉矶４月８日电（记者黄恒）美国第三舰队８日发布声明说，
该舰队下属的“卡尔·文森”航母战斗群当天离开新加坡，改变原定驶往澳大利亚
的任务计划，转而北上，前往西太平洋朝鲜半岛附近水域展开行动。\n　　
该舰队网站主页发布的消息说，美军太平洋司令部司令哈里·哈里斯指示“卡尔·文森”航母战斗群
向北航行。这一战斗群包括“卡尔·文森”号航空母舰、海军第二航空队、两艘“阿利·伯克”级导
弹驱逐舰和一艘“泰孔德罗加”级导弹巡洋舰。\n　　“卡尔·文森”号航母的母港位于美国加利福
尼亚州的圣迭戈，今年１月初前往西太平洋地区执行任务，并参与了日本及韩国的军事演习。\n　　
美国有线电视新闻网援引美国军方官员的话说，“‘卡尔·文森’号此次行动是为了对近期朝鲜的挑
衅行为作出回应”。（完）"""
#print(text)
import re
text = """美国有线电视新闻网援引美国军方官员的话说"""
pat = r'(新闻|官员)'#使用r前缀后，可以不用再转义
text_processed=re.compile(pat).sub(repl="**\g<1>**", string=text)#加** **

#print(text_processed)
#re.sub(pattern=模板, repl=替换成什么, string=, count=被替换的个数, flags= )

def get_query_pat(query):
    str=re.compile('({})'.format('|'.join(query.split())))
    return str
    #str.join(sequence),sequence是可迭代的对象，用str连接起来
#print(get_query_pat('美军 司令 航母'))
from IPython.display import display, Markdown
def highlight_keywords(pat, document):
    return pat.sub(repl="**\g<1>**", string=document)
print(highlight_keywords(get_query_pat('美军 司令 航母'), news_content['content'][1000]))

def search_engine_with_pretty_print(query):
    candidates_ids = search_engine(query)
    for i, _id in enumerate(candidates_ids):
        title = '## Search Result {}'.format(i)
        c = content['content'][_id]
        c = highlight_keywords(get_query_pat(query), c)
        display(Markdown(title + '\n' + c))

################################Page Rank

import random
from string import ascii_uppercase#ascii_uppercase是从A-Z的大写字母的字符串
#print(ascii_uppercase)

def genearte_random_website():
    return ''.join(
        [random.choice(ascii_uppercase) for _ in range(random.randint(3, 5))])+ '.' + random.choice(['com', 'cn', 'net'])
websites = [genearte_random_website() for _ in range(100)]
#print(websites)
print(random.sample(websites, 5))#websites中随便挑15个
website_connection = {
    websites[0]: random.sample(websites, 10),
    websites[1]: random.sample(websites, 5),
    websites[3]: random.sample(websites, 7),
    websites[4]: random.sample(websites, 2),
    websites[5]: random.sample(websites, 1),
}
import networkx as nx
website_network = nx.graph.Graph(website_connection)
nx.draw_networkx(website_network, font_size=10)
'''
import matplotlib.pyplot as plt
plt.figure(3,figsize=(12,12))
plt.show()
'''
#.items()：dict转换成list，[（key，value）].
# nx.pagerank(website_network)会得到website_network的pagerank值
print(sorted(nx.pagerank(website_network).items(),key=lambda x: x[1], reverse=True))