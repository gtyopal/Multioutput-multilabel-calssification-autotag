import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
import os

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join("__file__")))
AUTOTAG_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoints")
VOCAB_FILE = os.path.join(PARENT_DIR_PATH, "checkpoints","vocab_shape.pickle")
CONFIG_FILENAME = os.path.join(PARENT_DIR_PATH, 'config', 'config.ini')
YAML_FILE = os.path.join(PARENT_DIR_PATH, '','autotag.yml')
LOG_FILE = os.path.join(PARENT_DIR_PATH, 'logs', 'autotag')
TURI_RAW_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata', 'raw')
TURI_MODEL = os.path.join(PARENT_DIR_PATH, 'autotagmodel')
TURI_CLEAN_DATA = os.path.join(PARENT_DIR_PATH, 'autotagdata','cleaned')
W2V_MODEl = os.path.join(PARENT_DIR_PATH, "w2v_model")
WORD_PATH =  PARENT_DIR_PATH + "/init_data/word.pkl" 
EMBEDDING_PATH =  PARENT_DIR_PATH + "/init_data/embedding.pkl" 
DL_MODEL = PARENT_DIR_PATH + "/dl_model" 

MAXLEN = 600

word_file = open(WORD_PATH, 'rb') 
word_dict = pickle.load(word_file)
word_file.close()

label_dict_all = []
idx2label_all = []
for target_name in  ["level2","level3","level4","level5"]:
    LABEL_PATH =  PARENT_DIR_PATH + "/init_data/" +  target_name + "_label.pkl"
    label_file = open(LABEL_PATH, 'rb')
    idx2label = pickle.load(label_file)
    label_file.close()
    label_dict = {}
    for k,v in idx2label.items():
        label_dict[v] = k
    label_dict_all.append( label_dict )
    idx2label_all.append( idx2label ) 


def load_model(graph, target_name):
    with graph.as_default():
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        sess = tf.Session(config = sess_config,graph= graph)
        target = target_name + "_clean"
        mode = "cnn"
        MODEL_DIR = DL_MODEL + "/" + target + "/" + mode
        checkpoint_file = tf.train.latest_checkpoint(MODEL_DIR)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_data").outputs[0]
        input_x_len = graph.get_operation_by_name("input_data_len").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        prediction = graph.get_operation_by_name("acc_layer/pre_proba").outputs[0]

    return sess, input_x, input_x_len, keep_prob, prediction



def eval(sess, input_x, input_x_len, keep_prob, prediction, message, label_dict, idx2label):
    size = len(message)
    message2id = np.zeros((size, MAXLEN), dtype='int32')
    message_len =  np.zeros((size,), dtype='int32')
    label = np.zeros((size,len(label_dict)), dtype='int32')
    for index, line in enumerate(message):
        mess_tmp = line.strip().split()
        for k in range( min(MAXLEN, len(mess_tmp)) ): 
            message2id[index][k] = word_dict[ mess_tmp[k] ] if mess_tmp[k] in word_dict else 1
        message_len[index] = MAXLEN if len(mess_tmp) > MAXLEN else len(mess_tmp)
    x_prediction = sess.run(prediction, {input_x: message2id, input_x_len: message_len, keep_prob: 1.0 })
    result = []
    for i,rate in enumerate(x_prediction[0]):
        result.append((idx2label[i],rate))

    return result 

if __name__ == '__main__':

    message = ["learn how to sign in and place orders for apple 's print products such as books , calendars , and cards featuring your photos digit verify that your computer is connected to the internet , then open iphoto or aperture digit select a photo from your library by clicking it one time digit from the file menu , choose order prints digit click account info or set up account digit enter your apple id , which can be the same as your email address , but does n't have to be digit enter your password and click sign in and then click done digit your apple id will be displayed on the left side of the order window just above the shipping options menus digit click cancel \ ( your account info will still be saved \ ) digit you can use your icloud username and password anywhere an apple id is required digit & nbsp ; if your apple id is an email address , be sure to enter the complete address , such as & quot ; yourname @ me digit com digit & quot ; you can learn more about your apple id digit verify that your computer is connected to the internet , then open iphoto digit from the main iphoto window , click a project in the source list digit & nbsp ; if you are in the book shelf view , double click a project digit & nbsp ; choose buy card , buy book , or buy calendar at the bottom of the screen digit enter the quantity and select check out digit enter your apple id , which can be the same as your email address , but does n't have to be digit & nbsp ; enter your password , click sign in , and continue to place your order digit you can use your icloud username and password anywhere an apple id is required digit & nbsp ; if your apple id is an email address , be sure to enter the complete address , such as & quot ; yourname @ me digit com digit & quot ; & nbsp ; you can learn more about your apple id digit sign in and place an order from aperture or iphoto"]
    # message = input("Enter sentence, Ctrl+C to exit:")
    graph_all = [ tf.Graph() for i in range(4)]
    target_all = ["level2","level3","level4","level5"]
    sess_all = []
    input_x = []
    input_x_len = []
    keep_prob = []
    prediction = []
    for graph,target in zip(graph_all, target_all):
        graph_attri = load_model(graph, target)
        sess_all.append( graph_attri[0] )
        input_x.append( graph_attri[1] )
        input_x_len.append( graph_attri[2] )
        keep_prob.append( graph_attri[3] )
        prediction.append( graph_attri[4] )

    for mess in message:
        result_all = []
        print("message :",mess)
        for index in range( len(target_all) ):
            # predict label proba 
            result = eval(sess_all[index], input_x[index], input_x_len[index], keep_prob[index], prediction[index], [mess], label_dict_all[index], idx2label_all[index])
            #print(result)
            result_sort = sorted(result, key = lambda x: x[1], reverse = True)
            result_all.append(result_sort[0])
        print(result_all)
        print("-------------------------\n")
