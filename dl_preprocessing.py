# -*- coding: utf-8 -*-
import os
import turicreate as tc
import pandas as pd
import math
import numpy as np
from gensim.models import Word2Vec
import pickle

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join("__file__")))
SPAM_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoints")
VOCAB_FILE = os.path.join(PARENT_DIR_PATH, "checkpoints","vocab_shape.pickle")
CONFIG_FILENAME = os.path.join(PARENT_DIR_PATH, 'config', 'config.ini')
YAML_FILE = os.path.join(PARENT_DIR_PATH, '','autotag.yml')
LOG_FILE = os.path.join(PARENT_DIR_PATH, 'logs', 'autotag')
TURI_RAW_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata', 'raw')
TURI_MODEL = os.path.join(PARENT_DIR_PATH, 'autotagmodel')
TURI_CLEAN_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata','cleaned')
W2V_MODEl = os.path.join(PARENT_DIR_PATH, "w2v_model")
LABEL_PATH =  PARENT_DIR_PATH + "/init_data"
WORD_PATH =  PARENT_DIR_PATH + "/init_data/word.pkl" 
EMBEDDING_PATH =  PARENT_DIR_PATH + "/init_data/embedding.pkl" 

LANGUAGE_CODE = 'en'

target_label =  ['level2_clean','level3_clean','level4_clean','level5_clean']
target_raw = ['level2','level3','level4','level5']


RAW_DATA_FIlE = TURI_RAW_DATA+'/{}.csv'.format(LANGUAGE_CODE)
OUTPUT_DIR = TURI_CLEAN_DATA+'/{}/'.format(LANGUAGE_CODE)

print("Reading CSV file.")
pd_df = pd.read_csv(RAW_DATA_FIlE, encoding="utf8")

# Convert to SFrame
print("Converting to an SFrame.")
pd_df['level1'] = pd_df['level1'].fillna(value='none1')
pd_df['level2'] = pd_df['level2'].fillna(value='none2')
pd_df['level3'] = pd_df['level3'].fillna(value='none3')
pd_df['level4'] = pd_df['level4'].fillna(value='none4')
pd_df['level5'] = pd_df['level5'].fillna(value='none5')
pd_df['message'] = pd_df['message'].fillna(value='')
df = pd_df[pd_df["message"] !="" ]
oid = []
message = []
level1 = []
level2 = []
level3 = []
level4 = []
level5 = []
for idx in set(df['id'].tolist()):
    df_tmp = df[df["id"] == idx]
    oid.append(idx)
    mess = list( set(df_tmp["message"].tolist() ) )
    if len(mess) != 1:
        print("error")
    message.append(mess[0])
    level1.append( "|".join( list(set(df_tmp["level1"].tolist())) ) )
    level2.append( "|".join( list(set(df_tmp["level2"].tolist())) ) )
    level3.append( "|".join( list(set(df_tmp["level3"].tolist())) ) )
    level4.append( "|".join( list(set(df_tmp["level4"].tolist())) ) )
    level5.append( "|".join( list(set(df_tmp["level5"].tolist())) ) )
del df, pd_df

pd_df = pd.DataFrame({"id":oid,"level1":level1,"level2":level2,"level3":level3,"level4":level4,"level5":level5,"message":message})
def get_level_label(level_label, level_id):
    label_list = []
    label_tmp = level_label.strip().split("|")
    if len(label_tmp) == 1:
        return label_tmp
    else:
        for label in label_tmp:
            if "none" not in label:
                label_list.append(label)
        return label_list
for i in range(len(target_label)):
    pd_df[target_label[i]] = pd_df[target_raw[i]].apply(lambda x : get_level_label(x, 2) )    
       

# save all labels to pickle format 
from collections import Counter
for k in range(len(target_label)):
    tmp_dict = dict( Counter(sum(pd_df[target_label[k]].tolist(),[])) )
    tmp_sort = sorted( tmp_dict.items(), key = lambda x: x[1],reverse=True)
    label2idx = {}
    idx2label = {}
    for i,w in enumerate(tmp_sort):
        label2idx[w[0]] = i
        idx2label[i] = w[0]
    label_path = LABEL_PATH + "/" + target_raw[k] + "_label.pkl"
    file_w = open(label_path, 'wb')
    pickle.dump(idx2label, file_w)
    file_w.close()

sf = tc.SFrame(pd_df)
for i in range(len(target_label)):
    sf[target_label[i]] = sf[target_label[i]].apply(lambda x: "|".join(x))
del pd_df

#################
sf.save(OUTPUT_DIR + 'all-{}-data'.format(LANGUAGE_CODE))
train_set, test_set = sf.random_split(.9)

# Create the training
train_set.save(OUTPUT_DIR + 'train_set')

# Create the test set
test_set.save(OUTPUT_DIR + 'test_set')
assert(len(test_set) + len(train_set)) == len(sf)


# produce word2vec

vec_dim = 128
message_corpus = [line.strip().split() for line in sf["message"] ]
model  = Word2Vec(message_corpus, sg=1, size=vec_dim,  window=5,  min_count=3,  negative=3, sample=0.001, hs=1, workers=4)
model.wv.save_word2vec_format(W2V_MODEl + '/embedding.txt', binary=False)

# filter stop words
train_set["message_bow"] = tc.text_analytics.count_words(train_set['message'])
train_set["message_bow"] = tc.text_analytics.drop_words(train_set["message_bow"],
                                                           stop_words=tc.text_analytics.stop_words())                                                              
test_set["message_bow"] = tc.text_analytics.count_words(test_set['message'])
test_set["message_bow"] = tc.text_analytics.drop_words(test_set["message_bow"],
                                                           stop_words=tc.text_analytics.stop_words())
# save data to dataframe format
train_df = train_set.to_dataframe()
test_df = test_set.to_dataframe()
train_df.to_csv("train_set.csv",encoding = "utf8",index = False)
test_df.to_csv("test_set.csv",encoding = "utf8",index = False)

# create word_dict and save to pickle format
from collections import defaultdict
word_dict = defaultdict(int)
word_all = train_df["message_bow"].tolist() + test_df["message_bow"].tolist()
for words in word_all:
    for w,c in words.items():
        word_dict[w] += c
word_set = sorted([item[0] for item in word_dict.items()])
word_sort = {}
start = 2
for i, w in enumerate(word_set):
    idx = i + start
    word_sort[w] = idx
file_w = open(WORD_PATH, 'wb')
pickle.dump(word_sort, file_w)
file_w.close()

# save embedding to pickle format 
import codecs
file_r = codecs.open(W2V_MODEl + '/embedding.txt', 'r', encoding='utf-8')
line = file_r.readline()
word_size, embed_dim = map(int, line.split(' '))
embedding_dict = dict()
line = file_r.readline()
while line:
    embed = line.split(' ')
    word = embed[0]
    vec = np.array(embed[1:], dtype='float32')
    embedding_dict[word] = vec
    line = file_r.readline()
file_r.close()

embedding_size = len(word_sort) + start
embedding = np.zeros((embedding_size, vec_dim), dtype='float32')
for word in word_sort:
    if word in embedding_dict:
        embedding[word_sort[word], :] = embedding_dict[word]
    else:
        embedding[word_sort[word], :] = np.random.uniform(-0.25, 0.25, size=(embed_dim))      
embedding[1, :] = np.random.uniform(-0.25, 0.25, size=(vec_dim))

file_w = open(EMBEDDING_PATH, 'wb')
pickle.dump(embedding, file_w)
file_w.close()

