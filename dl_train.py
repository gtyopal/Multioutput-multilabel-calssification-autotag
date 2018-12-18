# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
import os
import math
import time
import codecs
import sys

target_name = sys.argv[1]
level_list = ["level2","level3","level4","level5"]
if target_name not in level_list:
    print("level name is not exit, please input the level_list:", level_list )
    exit()
mode_name = sys.argv[2]
mode_list = ["cnn","rnn_cnn"]
if mode_name not in mode_list:
    print("the current model has : ", mode_list)
    exit()

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

LABEL_PATH =  PARENT_DIR_PATH + "/init_data/" +  target_name + "_label.pkl" 
WORD_PATH =  PARENT_DIR_PATH + "/init_data/word.pkl" 
EMBEDDING_PATH =  PARENT_DIR_PATH + "/init_data/embedding.pkl" 
DL_MODEL = PARENT_DIR_PATH + "/dl_model" 
if not os.path.exists(DL_MODEL):
    os.makedirs(DL_MODEL)


MAXLEN = 600

word_file = open(WORD_PATH, 'rb') 
word_dict = pickle.load(word_file)
word_file.close()

label_file = open(LABEL_PATH, 'rb')
idx2label = pickle.load(label_file)
label_file.close()
label_dict = {}
for k,v in idx2label.items():
    label_dict[v] = k

embedding_file = open(EMBEDDING_PATH, 'rb') 
embedding_weight = pickle.load(embedding_file)
embedding_file.close()

target = target_name + "_clean"
mode = mode_name
MODEL_PATH = DL_MODEL + "/" + target


def generate_batch(input_x, input_x_len, input_y, batch_size, random_flag ):
        """
        Generates a batch iterator for a dataset.
        """
        input_size = len(input_x)
        total_batch = int(math.ceil(float(input_size/batch_size)))
        # Shuffle the data
        if random_flag == True:
            shuffle = np.random.permutation(np.arange(input_size) )
            input_x = input_x[shuffle]
            input_x_len =  input_x_len[shuffle] 
            input_y = input_y[shuffle]

        for batch_num in range(total_batch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, input_size)
            batch_x = input_x[start:end]
            batch_x_len =  input_x_len[start:end]
            batch_y = input_y[start:end]
            yield (batch_x, batch_x_len, batch_y)

# create model
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    session = tf.Session(config=config)
    ##with sess.as_default():

    embedded_size = embedding_weight.shape[1]
    dense_size = 200
    rnn_size = 200
    
    input_x = tf.placeholder(tf.int32, shape = (None, MAXLEN), name="input_data")
    input_x_len = tf.placeholder(tf.int32, shape = [None,], name='input_data_len')
    label_size = len(label_dict)
    input_y = tf.placeholder(tf.int32, shape = [None, label_size], name="input_label")
    dropout_prob = tf.placeholder(tf.float32, name="keep_prob")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    with tf.variable_scope("embedding_layer"):
        word_embedding = tf.get_variable('embedding', shape= embedding_weight.shape, initializer= tf.constant_initializer(value = embedding_weight) ) 
        print( word_embedding.get_shape().as_list() )
        input_embed_raw = tf.nn.embedding_lookup(word_embedding, input_x)
        input_embed = tf.nn.dropout(input_embed_raw, dropout_prob, name="input_dropout")

    hidden_ouput = ""
    with tf.variable_scope("inter_layer"):
        if "rnn" in mode:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=2.0), output_keep_prob=dropout_prob )
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(rnn_size, forget_bias=2.0), output_keep_prob=dropout_prob )

            #fw_cells =tf.nn.rnn_cell.MultiRNNCell(   \
            #tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell( unit_size), output_keep_prob=self.lstm_dropout_keep_prob), \
            #tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell( unit_size), output_keep_prob=self.lstm_dropout_keep_prob)  )
            (fw_output, bw_output), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn( 
                                            fw_cell, bw_cell, input_embed, sequence_length=input_x_len, dtype=tf.float32 )  
            rnn_output = tf.concat( (fw_state.h, bw_state.h), axis = -1, name = "concat_state")
            hidden_ouput = rnn_output

            if mode == "rnn_cnn":
                birnn_output = tf.concat((fw_output, bw_output), axis = -1, name = "concat_birnn_output")
                cnn_input = tf.expand_dims(birnn_output, -1)
                num_filters = [128,128,128,128,128] 
                filter_sizes = [1,2,3,4,5]
                cnn_output = []
                cnn_size = rnn_size * 2
                for i,filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        filter_shape = [filter_size,cnn_size,1,num_filters[i]]
                        cnn_weight1 = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "cnn_weith1")
                        cnn_bias1 = tf.Variable(tf.constant(0.1,shape=[num_filters[i]]), name = "cnn_bias1")

                        input_conv = tf.nn.conv2d(cnn_input, cnn_weight1, strides=[1, 1, 1, 1], padding = "VALID",name = "cnn_conv1")
                        input_func = tf.nn.relu(tf.add(input_conv, cnn_bias1 ), name = "cnn_func1")
                        input_pooled = tf.reduce_max(tf.squeeze(input_func, [2]), 1)
                        cnn_output.append(input_pooled)

                filters_total = sum(num_filters)
                cnn_output_flat = tf.reshape(tf.concat(values = cnn_output, axis = 1), [-1, filters_total])
                cnn_output_drop = tf.nn.dropout(cnn_output_flat, dropout_prob)
                hidden_ouput = cnn_output_drop

        elif mode == "cnn":
            cnn_input = tf.expand_dims(input_embed, -1)
            num_filters = [256,256,256,256,256] 
            filter_sizes = [1,2,3,4,5]
            cnn_output = []
            for i,filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size,embedded_size,1,num_filters[i]]
                    cnn_weight1 = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "cnn_weith1")
                    cnn_bias1 = tf.Variable(tf.constant(0.1,shape=[num_filters[i]]), name = "cnn_bias1")

                    input_conv = tf.nn.conv2d(cnn_input, cnn_weight1, strides=[1, 1, 1, 1], padding = "VALID",name = "cnn_conv1")
                    input_func = tf.nn.relu(tf.add(input_conv, cnn_bias1 ), name = "cnn_func1")
                    input_pooled = tf.reduce_max(tf.squeeze(input_func, [2]), 1)
                    cnn_output.append(input_pooled)

            filters_total = sum(num_filters)
            cnn_output_flat = tf.reshape(tf.concat(values = cnn_output, axis = 1), [-1, filters_total])
            cnn_output_drop = tf.nn.dropout(cnn_output_flat, dropout_prob)
            hidden_ouput = cnn_output_drop
            
    with tf.variable_scope("output_layer"):     
        dense_output= tf.layers.dense(hidden_ouput, dense_size * 2, activation=tf.nn.relu, name="dense_layer")
        dense_output = tf.nn.dropout(dense_output, dropout_prob)
        softmax_weight = tf.get_variable("weight", [dense_size * 2, label_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        softmax_bias = tf.get_variable("bias", [label_size], dtype=tf.float32)
        output = tf.add(tf.matmul(dense_output, softmax_weight), softmax_bias, name="scores")

    with tf.name_scope("loss_layer"): 
        input_y = tf.cast(input_y, tf.float32)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = output,labels = input_y,name='multi_entropy')
        loss = tf.reduce_mean(cross_entropy, name='entropy_mean')
    
    with tf.name_scope("acc_layer"): 
        pre_proba =tf.sigmoid(output, name = "pre_proba")
        pre_y = tf.cast(tf.greater_equal(pre_proba, 0.5), tf.int32)
        pre_equal = tf.cast( tf.equal(pre_y, tf.cast(input_y,tf.int32)), tf.int32)
        prediction = tf.reduce_mean(pre_equal,axis = 1)
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"), name="accuracy")

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9,beta2=0.999,epsilon=1e-08)
    train_vars = tf.trainable_variables()
    if "rnn" in mode:
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 5)
        train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=global_step)
    elif (mode == "cnn") or (mode == "cnn_merge") or (mode == "3layer_cnn"):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    '''''''''''''''''''''''''''''''''''''''''''''
    TRAIN
    '''''''''''''''''''''''''''''''''''''''''''''
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep= 1)

    base_path = os.path.abspath(os.path.join(MODEL_PATH, mode) )
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    checkpoint_prefix = os.path.join(base_path, "model")

    def train():
        # pross train_data to sequence id
        train_df = pd.read_csv("train_set.csv", encoding = "utf8")
        train_data = train_df['message'].tolist()
        train_label_raw = train_df[target].tolist()
        train_size = train_df.shape[0]
        train_message = np.zeros((train_size, MAXLEN), dtype='int32')
        train_message_len =  np.zeros((train_size,), dtype='int32')
        train_label = np.zeros((train_size,len(label_dict)), dtype='int32')
        for index, line in enumerate(train_data):
            mess_tmp = line.strip().split()
            for k in range( min(MAXLEN, len(mess_tmp)) ): 
                train_message[index][k] = word_dict[ mess_tmp[k] ] if mess_tmp[k] in word_dict else 1
            train_message_len[index] = MAXLEN if len(mess_tmp) > MAXLEN else len(mess_tmp)
            for label_name in train_label_raw[index].strip().split("|"):
                try:
                    train_label[index, label_dict[label_name] ] = 1
                except:
                    print(train_label_raw[index])

        # pross test_data to sequence id
        test_df = pd.read_csv("test_set.csv", encoding = "utf8")
        test_data = test_df['message'].tolist()
        test_label_raw = test_df[target].tolist()
        test_size = test_df.shape[0]
        test_message = np.zeros((test_size, MAXLEN), dtype='int32')
        test_message_len =  np.zeros((test_size,), dtype='int32')
        test_label = np.zeros((test_size,len(label_dict)), dtype='int32')
        for index, line in enumerate(test_data):
            mess_tmp = line.strip().split()
            for k in range( min(MAXLEN, len(mess_tmp)) ): 
                test_message[index][k] = word_dict[ mess_tmp[k] ] if mess_tmp[k] in word_dict else 1
            test_message_len[index] = MAXLEN if len(mess_tmp) > MAXLEN else len(mess_tmp)
            for label_name in test_label_raw[index].strip().split("|"):
                test_label[index, label_dict[label_name] ] = 1

        # save training result to file 
        best_accuracy = 0
        patient = 0
        f_train = codecs.open(base_path + "/train_acc_loss.txt","w",encoding = "utf8")
        f_train.write("step \tloss \taccuracy\n")
        f_test = codecs.open(base_path +  "/test_acc_loss.txt","w",encoding = "utf8")
        f_test.write("step \tloss \taccuracy\n")

        # train model
        train_epoch = 50000
        batch_size = 64
        learn_rate = 0.001
        for epoch in range(train_epoch):
            training_batches = generate_batch(train_message, train_message_len, train_label, batch_size, random_flag = True)
            print ('epoch {}'.format(epoch + 1))
            for batch_x, batch_x_len, batch_y in training_batches:
                time_start = time.time()
                feed_dict = dict()
                feed_dict[input_x] = batch_x
                feed_dict[input_x_len] = batch_x_len
                feed_dict[input_y] = batch_y
                feed_dict[dropout_prob] = 0.8
                feed_dict[learning_rate] = learn_rate
                fetches = [train_op, global_step, loss, accuracy]
                _, global_step_tmp, train_loss, train_accuracy = session.run(fetches, feed_dict)
                step_time = time.time() - time_start
                sample_psec = batch_size / step_time
                print ("Train, step {}, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}".format(global_step_tmp, train_loss, train_accuracy, step_time, sample_psec))
                f_train.write(str(global_step_tmp) + "\t" + str(train_loss) + "\t" + str(train_accuracy) + "\n")
            f_train.flush()

            # evaluate_test_set
            test_time = time.time()
            testing_batches = generate_batch(test_message, test_message_len, test_label, batch_size, random_flag = False)
            test_loss_all = []
            test_prediction_all = []
            for batch_x, batch_x_len, batch_y in testing_batches:
                feed_dict = dict()
                feed_dict[input_x] = batch_x
                feed_dict[input_x_len] = batch_x_len
                feed_dict[input_y] = batch_y
                feed_dict[dropout_prob] = 1
                fetches = [loss, prediction]
                test_loss, test_prediction = session.run(fetches, feed_dict)
                test_loss_all.append(test_loss)
                test_prediction_all.append(test_prediction)
            step_time = time.time() - test_time
            sample_psec = len(test_message) / step_time
            tess_loss = np.mean(test_loss_all)
            test_accuracy = np.mean(np.concatenate(test_prediction_all))
            print ("Test, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}".format(test_loss, test_accuracy, step_time, sample_psec))
            f_test.write(str(global_step_tmp) + "\t" + str(test_loss) + "\t" + str(test_accuracy) + "\n")
            f_test.flush()
            if test_accuracy > best_accuracy:
                patient = 0
                best_accuracy = test_accuracy
                model_path = saver.save(session, checkpoint_prefix, global_step=global_step_tmp)
                print("Saved model checkpoint to {}\n".format(model_path))
                # write out 3 files
                saver.save(session, base_path+'/trained_{}_model.sd'.format(mode_name))
                tf.train.write_graph(session.graph_def, '.', base_path+'/trained_{}_model.proto'.format(mode_name), as_text=False)
                tf.train.write_graph(session.graph_def, '.', base_path+'/trained_{}_model.txt'.format(mode_name), as_text=True)
            else:
                patient += 1
                print("Not improved for continuous num of epochs of ",patient, "Best accuracy is ", best_accuracy, "Learning_rate is", learn_rate)
                if patient > 3:
                    learn_rate = learn_rate / 5.0
            if patient > 10:
                # Use a saver_def to get the "magic" strings to restore
                saver_def = saver.as_saver_def()
                # The name of the tensor you must feed with a filename when saving/restoring.
                print ("The name of the tensor you must feed with a filename when saving/restoring: ",saver_def.filename_tensor_name)
                # The name of the target operation you must run when restoring.
                print ("The name of the target operation you must run when restoring: ",saver_def.restore_op_name)
                # The name of the target operation you must run when saving.
                print ("The name of the target operation you must run when saving: ", saver_def.save_tensor_name)
                print("Not improved for continuous num of epochs of ", patient, "Best accuracy is ", best_accuracy, "Learning_rate is", learn_rate)
                print("Accuracy has not improved at {} time , training model done!".format(patient))
                break
                
    train()