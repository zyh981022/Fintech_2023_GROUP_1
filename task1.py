import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import warnings
import time
from tqdm import tqdm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import category_encoders as ce
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import re
import jieba
import nltk
from nltk import pos_tag
from gensim.models.keyedvectors import KeyedVectors
import time
from sklearn.linear_model import LogisticRegression

data = pd.read_excel('D:/招行训练营/数据集/科创企业行业先进性判断子模型/data/工商信息-经营范围.xlsx')
def match_word(text):
    pattern = r'[\u4e00-\u9fff]+'  # 匹配中文字符的 Unicode 范围
    return  re.findall(pattern, text)
data['经营范围'] = data['经营范围'].astype(str).fillna('')
data['经营范围分词'] = data['经营范围'].apply(match_word)

def filter_word(word_ls):

    # 标注词性
    seg_list = jieba.lcut(''.join(word_ls))
    tagged_words = pos_tag(seg_list)
    pos_to_remove = ['v', 'p', 'a']

    # 过滤非动词
    filtered_words = [word for word, pos in tagged_words if pos[0] not in  pos_to_remove and "一般经营项目" not in word and"一般项目" not in word and word!='一般' and word!='项目']

    return filtered_words
data ['过滤分词'] = data['经营范围分词'].apply(filter_word)
model = KeyedVectors.load_word2vec_format('D:\中文语料\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5', binary=False, unicode_errors='ignore')
ninth_keywords = ['新一代信息技术', '新能源汽车', '高端装备', '新能源', '新材料', '绿色环保', '生物技术', '航空航天', '海洋装备']
ninth_keywords_dict = {
    '新一代信息技术': ['信息技术','科技革命','数字化','创新信息','现代信息','下一代信息','高新信息','新兴信息','前沿信息'],
    '新能源汽车':['电动汽车', '混合动力汽车', '绿色汽车', '清洁能源汽车', '可再生能源汽车', '零排放汽车', '非传统能源汽车', '高效能源汽车', '新型能源汽车', '低碳汽车'],
    '高端装备':['先进装备', '尖端装备', '先进设备', '高级设备', '高科技装备', '高级装置', '高级机械', '高科技设备', '高技术装备', '先进工具'],
    '新能源':['可再生', '清洁', '可持续', '替代', '绿色', '未来', '可更新', '低碳', '环保', '非化石'],
    '新材料':['先进材料', '创新材料', '高新材料', '新型材料', '新颖材料', '先进材质', '新材质', '新科技材料', '高性能材料', '先进复合材料'],
    '绿色环保':['可持续发展', '生态友好', '环境友好', '低碳环保', '可循环利用', '清洁环保', '可再生', '资源节约', '绿色可持续', '生态保护'],
    '生物技术':['生命科技', '生物工程', '生物科技', '生物工艺', '生物医药', '生物学技术', '生物制造', '生物创新', '基因工程', '分子生物技术'],
    '航空航天':['航天', '航空', '宇航','发动机','航发','轴承','机床','高速','复合材料','推进剂','动力'],
    '海洋装备':['海洋', '工装', '资源开发', '石油','船舶','钻井','深海','潜水','深水']
}
label =pd.read_csv(r'D:\招行训练营\数据集\科创企业行业先进性判断子模型\label\train_1.csv')
label.rename(columns={'label':'Y'},inplace=True)
test_df = pd.read_csv(r'D:\招行训练营\数据集\科创企业行业先进性判断子模型\upload.csv')
df = pd.concat((label,test_df))
data.rename(columns={'企业id':'ID'},inplace=True)
data.drop_duplicates(subset='ID', keep='first', inplace=True)
df = df.merge(data,how='left',on="ID")
features = pd.DataFrame(columns=ninth_keywords_dict.keys())
features['ID'] = df["ID"].copy()
def calculate_similarity(word_ls,indurstry):
    max_score = 0
    for i,word in enumerate(word_ls):
        #print(i)
        for item in ninth_keywords_dict[indurstry]:
            word1 = item
            word2 = word
            #print(word1,word2)
            try:
                similar_score = model.similarity(word1, word2)
                if similar_score>max_score:
                    max_score = similar_score
            except:
                pass
    return max_score
features['过滤分词'] = df['过滤分词']
for key in ninth_keywords_dict.keys():
    indurstry = key
    features[key] = data['过滤分词'].apply(lambda txt:calculate_similarity(txt,key))
guobiao  = pd.read_excel('D:/招行训练营/数据集/科创企业行业先进性判断子模型/data/工商信息-企业规模国标.xlsx')
guobiao ['企业规模国标'] = guobiao ['企业规模国标'].replace({'未获取':0,'微型':1,'小型':2,'中型':3,'大型':4})
guobiao.rename(columns={'企业id':'ID'},inplace=True)
features = features.merge(guobiao,how='left',on='ID')
fea_columns = list(ninth_keywords_dict.keys())+ ['企业规模国标_x']
features[list(fea_columns)]= features[fea_columns].fillna(0)
features['Y']= df['Y']
train_X = features[~features['Y'].isnull()][fea_columns]
train_Y=  features[~features['Y'].isnull()]['Y']
test_X = features[features['Y'].isnull()][fea_columns]
model = LogisticRegression()
model.fit(train_X, train_Y)
predictions = model.predict_proba(test_X)[:,1]
percentile_80 = np.percentile(predictions, 80)
sub_data = features[features['Y'].isnull()]
sub_data['score'] = predictions
sub_data['Y'] = 1*(predictions>percentile_80)
submit_data = sub_data[['ID','score','Y']]
submit_data.to_csv('./upload1.csv', encoding='utf-8-sig', index=False)