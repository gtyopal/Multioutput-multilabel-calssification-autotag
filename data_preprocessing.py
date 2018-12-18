# coding: utf-8
import numpy as np
import json

import multiprocessing
from multiprocessing import Pool
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
import pandas as pd

def tag_split(data_list):
    ''' add at 20180922 '''
    res = []
    for line in data_list:
        try:
            oid,tags,message = line[0],line[1],line[2]
            tag_list = ['' for i in range(5)]
            fields = tags.split(r"/")
            for i in range(len(fields)):
#                tag_list[i] = fields[i]
                if i == 0:
                    tag_list[i]=fields[i]
                else:
                    tag_list[i] = '/'.join([tag_list[i-1],fields[i]])
            line_res = [oid]
            line_res.extend(tag_list)
            line_res.append(message)
            res.append(line_res)
        except:
            continue
    return res

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    """ remove url """
    string = re.sub(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
                    ' spamurl ', string)
    """ remove email """
    string = re.sub(r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)', ' email ', string)
    """ remove phone numbers """
    string = re.sub(r'[\@\+\*].?[014789][0-9\+\-\.\~\(\) ]+.{6,}', ' phone ', string)
    """ remove digits """
    string = re.sub(r'[0-9\.\%]+', ' digit ', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.encode('utf-8').strip().lower()

def strip_html(string):
    soup = BeautifulSoup(string, "html.parser")
    string = soup.get_text()
    r = re.compile(r'<[^>]+>', re.S)
    string = r.sub('', string)
    string = re.sub(r'&(nbsp;)', ' ', string)
    string = re.sub(r'<[^>]+', '', string)
    string = re.sub('\&lt[;]', ' ', string)
    string = re.sub('\&gt[;]', ' ', string)
    return string

def denoise_text(string):
    string = clean_str(string)
    string = strip_html(string)
    words = word_tokenize(string)
    string = " ".join(words)
    if not string.strip():
        string = " "
    return string

def task(texts):
    res = []
    for line in texts:
        t = denoise_text(line[1])
        if len(t.split()) > 0:
            res.append([line[0], t, line[2]])
    return res

def remove_non_ascii(x):
    words = word_tokenize(x[1])
    new_words = []
    for word in words:
        if all(ord(c) < 128 for c in word):
            new_words.append(word)
    return [x[0], ' '.join(new_words), x[2]]

def clean(data_path):
    with open(data_path, "r",encoding='utf-8') as fr:
        data = fr.readlines()
    
    oid_contents = []

    for line in data:
        try:
            temp = json.loads(line)

            if not (temp["articleId"].startswith("HT") and
                    temp["localeName"].lower() == "english"):
                continue
    
            oid = temp["_id"]["$oid"]
            articleId = temp["articleId"]
            main = temp["content"]["main"]
            sections = temp["content"]["sections"]
            tags = temp["tags"]
            oid_contents.append(["%s-%s" % (articleId, oid ), main, sections, tags])

        except Exception:
            pass   
    print(len(oid_contents))

    all_texts = []
    for i in range(len(oid_contents)):
        oid = oid_contents[i][0]
        
        main = oid_contents[i][1]
        sections = oid_contents[i][2]
        summary, text, title, communities = '', '', '', []
        if "summary" in main.keys():
            summary = main["summary"]
            if not isinstance(summary, str):
                summary = summary["summaryText"]

        if "title" in main.keys():
            title = main["title"]

        for s in sections:
            if "text" in s.keys():
                text = ' '.join([text, s["text"]])

        tags = oid_contents[i][3]

        communities = [tag["tagPath"] for tag in tags if tag["type"] == "productcategory"]
        all_texts.append([oid, ' '.join([summary, text, title]), communities])

    all_texts = list(map(remove_non_ascii, all_texts))

    print("english texts number: ", len(all_texts))

    core_num = multiprocessing.cpu_count()
    print("core_num:", core_num)
    slices = np.linspace(0, len(all_texts), core_num+1).astype(int)

    pool = Pool(core_num)
    temp = []

    for i in range(core_num):
        slice_texts = all_texts[slices[i]:slices[i+1]]
        temp.append(pool.apply_async(task, args=(slice_texts,)))
    pool.close()
    pool.join()

    print("end.....")

    result = []
    for t in temp:
        result.extend(t.get())
        
    ''' add at 20180921 '''
    res = []
    for item in result:
        if len(item) != 3:
            continue
        oid, text, tag_list = item[0], item[1], item[2]
        mid_res = [[oid,tag_list[i],text] for i in range(len(tag_list))]
#        result.extend(["\t".join(mid_res_item) for mid_res_item in mid_res])
        res.extend(mid_res)
    res_tags_split = tag_split(res)
    df = pd.DataFrame(res_tags_split, columns = ['id','tag1','tag2','tag3','tag4','tag5','message'])
    df.to_csv(r"../autotagdata/raw/en_multilevel.csv",index=None)

if __name__ == "__main__":
    df = clean("../autotagdata/raw/0_r.txt")