# -*- coding: utf-8 -*-
import pandas as pd 
import os
import codecs

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join("__file__")))

DL_MODEL = PARENT_DIR_PATH + "/dl_model" 

target_name_list = ["level2","level3","level4","level5"]
mode_name_list =  ["cnn","rnn_cnn"]

result = []
for target_name in target_name_list:
	tmp = []
	for mode_name in mode_name_list:
		target = target_name + "_clean"
		mode = mode_name
		path = DL_MODEL + "/" + target + "/" + mode + "/"+ "test_acc_loss.txt"
		if not os.path.exists(path):
			tmp.append("None")
			continue
		f_r = codecs.open(path, encoding = "utf8")
		acc = []
		for line in f_r.readlines()[1:]:
			acc.append( float(line.strip().split()[2]) )
		if len(acc) == 0:
			tmp.append("None")
		else :
			tmp.append( max(acc) )
	result.append(tmp)

result_dict = {}
for i,name in enumerate(target_name_list):
	result_dict[name] = result[i]

result_df = pd.DataFrame(result_dict, index = mode_name_list )
print(result_df)
result_df.to_csv("dl_model_compare.csv", encoding = "utf8")



