import numpy as np
import tensorflow as tf
from random import shuffle
from datetime import datetime
from tensorflow.python.layers.core import Dense
# from utils.lstm import BNLSTMCell
import pickle
import math
import pdb
import os

class DeepCNNBLSTMSegment():
    def __init__(self, n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
                character_dictionary, character_embedding_size, max_word_length, filter_sizes, num_filters, \
                initial_type, dict_order_size, num_sampled):
        #    initialize session
        self.sess = tf.InteractiveSession()
        #    parameters for inputs
        self.word_dictionary = word_dictionary
        self.word_embedding_size = word_embedding_size
        self.character_dictionary = character_dictionary
        self.character_embedding_size = character_embedding_size
        self.dict_order_size = dict_order_size
        #    initial type = [xavier, normal]
        self.initial_type = initial_type
        #    parameters for CNN layers
        self.max_word_length = max_word_length
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        #    parameters for LSTM layers
        self.n_lstm_hidden = n_lstm_hidden
        self.max_lstm_step = max_lstm_step
        self.num_lstm_layers = num_lstm_layers
        #    parameters for syllable model output
        self.num_sampled = num_sampled
        #    parameters for outputs
        self.num_classes = 4
        self.build_model()

    def build_model(self):
        self.init_placeholder()
        self.build_graph()
        self.loss_optimizer()
        self.sess.run(tf.global_variables_initializer())
        # summary_writer = tf.train.SummaryWriter('/home/tittit/python/web_mining2/logs', graph = tf.get_default_graph())

    def init_placeholder(self):
        ### placeholders
        self.word_inputs = tf.placeholder(tf.int32, [None, self.max_lstm_step])
        self.char_inputs = tf.placeholder(tf.int32, [None, self.max_lstm_step, self.max_word_length])
        self.dict_order_feats = tf.placeholder(tf.float32, [None, self.max_lstm_step, self.dict_order_size])
        self.y = tf.placeholder(tf.int32, [None, self.max_lstm_step])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.lr = tf.placeholder(tf.float32, None)
        self.dropout = tf.placeholder(tf.float32, None)
        self.is_training = tf.placeholder(tf.bool, None)
        self.current_batch_size = tf.shape(self.word_inputs)[0]
        self.syllable_model_index = tf.placeholder(tf.int32, [None])
        self.syllable_model_label = tf.placeholder(tf.int32, [None])
        self.one_hot_matrix = tf.Variable(tf.eye(len(self.word_dictionary)), trainable = False)

    def build_graph(self):
        #    define type of initial variables
        if self.initial_type == "normal":
            #    initialize encoder embedding have variance = 1
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype = tf.float32)
        elif self.initial_type == "xavier":
            initializer = tf.contrib.layers.xavier_initializer()

        ### word embedding layer
        with tf.variable_scope('word_embedding_layer'):
            # word_embedding_matrix = tf.get_variable(name = 'word_embedding_matrix', shape = [len(self.word_dictionary), self.word_embedding_size], initializer = initializer, dtype = tf.float32)
            word_embedding_matrix = tf.get_variable(name = 'word_embedding_matrix', initializer = tf.random_uniform([len(self.word_dictionary), self.word_embedding_size], -1.0, 1.0))
            #    word inputs embedding: [batch size, max_lstm_step, embedding size]
            word_inputs_embedding = tf.nn.embedding_lookup(params = word_embedding_matrix, ids = self.word_inputs)
            # #    syllable vectors model training
            # syllable_model_input  = tf.nn.embedding_lookup(params = word_embedding_matrix, ids = self.syllable_model_index)
            # syllable_model_output = tf.reshape(self.syllable_model_label, (-1, 1))
            # nce_weights = tf.get_variable(name = 'nce_weights', shape = [len(self.word_dictionary), self.word_embedding_size], initializer = initializer, dtype = tf.float32)
            # nce_biases = tf.get_variable(name = 'nce_biases', shape = [len(self.word_dictionary)], initializer = initializer, dtype = tf.float32)
            # self.syllable_model_loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights, biases = nce_biases, labels = syllable_model_output, inputs = syllable_model_input, num_sampled = self.num_sampled, num_classes = len(self.word_dictionary)))
            # self.syllable_model_optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.polynomial_decay(learning_rate = 1.0, global_step = 0, decay_steps = 100, end_learning_rate = 0.1, power = 0.5)).minimize(self.syllable_model_loss)

        ### character embedding layer
        with tf.name_scope('character_embedding_layer'):
            character_embedding_matrix = tf.get_variable(name = 'character_embedding_matrix', shape = [len(self.character_dictionary), self.character_embedding_size], initializer = initializer, dtype = tf.float32)
            character_inputs_embedding = tf.nn.embedding_lookup(params = character_embedding_matrix, ids = self.char_inputs)
        
        ### CNN layer
        with tf.variable_scope("CNN_layer"):
            cnn_feats = []
            #    reshape character input embedding to [batch size*max_lstm_step, max_word_length, embedding size, 1]
            images = tf.reshape(character_inputs_embedding, [self.current_batch_size*self.max_lstm_step, self.max_word_length, self.character_embedding_size, 1])
            for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters)):
                w = tf.get_variable(shape = [filter_size, self.character_embedding_size, 1, num_filter], name = "w_conv_" + str(i), initializer = initializer, dtype = tf.float32)
                b = tf.get_variable(shape = [num_filter], name = "b_conv_" + str(i), initializer = initializer, dtype = tf.float32)
                #    conv has shape = [batchsize*max_lstm_step, k, 1, num_filter]
                conv = tf.nn.relu(tf.nn.conv2d(images, w, strides = [1, 1, 1, 1], padding = 'VALID'))
                #    conv_pool has shape [batchsize*max_lstm_step, 1, 1, num_filter]
                conv_pool = tf.reduce_max(conv, axis = 1)
                #    conv_feat has shape [batsize, max_lstm_step, num_filter]
                conv_feat = tf.reshape(conv_pool, [self.current_batch_size, self.max_lstm_step, num_filter])
                cnn_feats.append(conv_feat)
            output_cnn = cnn_feats[0]
            for cnn_feat in cnn_feats[1:]:
                output_cnn = tf.concat([output_cnn, cnn_feat], axis = -1)

        ### LSTM layer
        with tf.variable_scope("biLSTM_layer"):
            output = tf.concat([word_inputs_embedding, output_cnn, self.dict_order_feats], axis = -1)
            lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_lstm_hidden), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
            lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_lstm_hidden), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
            # lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(BNLSTMCell(num_units = self.n_lstm_hidden, is_training_tensor = self.is_training, max_bn_steps = self.max_lstm_step, dtype = tf.float32), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
            # lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(BNLSTMCell(num_units = self.n_lstm_hidden, is_training_tensor = self.is_training, max_bn_steps = self.max_lstm_step, dtype = tf.float32), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
            for i in range(self.num_lstm_layers):
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cells_fw[i], cell_bw = lstm_cells_bw[i], inputs = output, sequence_length = self.sequence_length, scope = 'B_LSTM_layer_' + str(i), dtype = tf.float32)
                output = tf.concat([output_fw, output_bw], axis = -1)
        
        with tf.variable_scope("fully_connected_layer"):
            w = tf.get_variable(shape = [2*self.n_lstm_hidden, self.num_classes], name = 'w', initializer = initializer, dtype = tf.float32)
            b = tf.get_variable(shape = [1, self.num_classes], name = 'b', initializer = initializer, dtype = tf.float32)
            output_slice = tf.reshape(output, [-1, 2*self.n_lstm_hidden])
            pred_slice = tf.matmul(output_slice, w) + b
            self.pred = tf.reshape(pred_slice, [self.current_batch_size, self.max_lstm_step, self.num_classes])

        ### CRF layer
        with tf.variable_scope("Linear_CRF_layer"):
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.pred, self.y, self.sequence_length)
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.pred, self.transition_params, self.sequence_length)

    def loss_optimizer(self):
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def shuffle(self, word_inputs, char_inputs, dict_order_feats, y, seqlen):
        z = list(zip(word_inputs, char_inputs, dict_order_feats, y, seqlen))
        shuffle(z)
        word_inputs, char_inputs, dict_order_feats, y, seqlen = zip(*z)
        return word_inputs, char_inputs, dict_order_feats, y, seqlen

    def evaluate(self, word_inputs_test, char_inputs_test, dict_order_feats, y_test, seqlen_test, batch_size):
        start = 0
        cnt = 0
        sum_cnt = np.sum(seqlen_test)
        while (start < len(seqlen_test)):
            batch_word_inputs_data = word_inputs_test[start:start + batch_size]
            batch_char_inputs_data = char_inputs_test[start:start + batch_size]
            batch_dict_order_feats = dict_order_feats[start:start + batch_size]
            batch_labels = y_test[start:start + batch_size]
            batch_seqlen = seqlen_test[start:start + batch_size]
            start += batch_size
            _pred = self.sess.run(self.viterbi_sequence, feed_dict = {self.word_inputs: batch_word_inputs_data, self.char_inputs: batch_char_inputs_data, self.dict_order_feats: batch_dict_order_feats, self.sequence_length: batch_seqlen, self.dropout: 1.0, self.is_training: False})
            for i in range(len(batch_seqlen)):
                cnt += np.sum(np.equal(_pred[i][:batch_seqlen[i]], batch_labels[i][:batch_seqlen[i]]))
        return cnt/sum_cnt

    def evaluate_f1(self, word_inputs_test, char_inputs_test, dict_order_feats, y_test, seqlen_test, batch_size):
        start = 0
        cnt = 0
        recall_cnt = 0
        precision_cnt = 0
        while (start < len(seqlen_test)):
            batch_word_inputs_data = word_inputs_test[start:start + batch_size]
            batch_char_inputs_data = char_inputs_test[start:start + batch_size]
            batch_dict_order_feats = dict_order_feats[start:start + batch_size]
            batch_labels = y_test[start:start + batch_size]
            batch_seqlen = seqlen_test[start:start + batch_size]
            start += batch_size
            _pred = self.sess.run(self.viterbi_sequence, feed_dict = {self.word_inputs: batch_word_inputs_data, self.char_inputs: batch_char_inputs_data, self.dict_order_feats: batch_dict_order_feats, self.sequence_length: batch_seqlen, self.dropout: 1.0, self.is_training: False})
            for i in range(len(batch_seqlen)):
                precision_cnt += list(_pred[i][:batch_seqlen[i]]).count(1)
                recall_cnt += list(batch_labels[i][:batch_seqlen[i]]).count(1)
                s = 0
                e = 0
                true_labels = batch_labels[i]
                pred_labels = _pred[i]
                while (e < batch_seqlen[i]):
                    e += 1
                    if (e == batch_seqlen[i]) or (true_labels[e] == 1):
                        if (list(true_labels[s:e]) == list(pred_labels[s:e])):
                            cnt += 1
                        s = e
        precision = cnt/precision_cnt
        recall = cnt/recall_cnt
        f1 = 2*precision*recall/(precision + recall)
        return precision, recall, f1

    def train_new_model(self, word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train, word_inputs_valid, char_inputs_valid, dict_order_feats_valid, y_valid, seqlen_valid, lr, dropout, batch_size, num_epochs, save_path, syllable_model_train_index, syllable_model_train_label):
        accuracy = 0
        saver = tf.train.Saver(max_to_keep = 1000)
        f = open(os.path.join(save_path, "log.txt"), "a")
        for epoch in range(num_epochs):
            ### shuffle all training data
            word_inputs, char_inputs, dict_order_feats, y, seqlen = self.shuffle(word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train)
            sum_loss_main = 0
            sum_loss_syl  = 0
            start_time = datetime.now()
            start = 0
            while start < len(seqlen):
                ### get training batch
                word_inputs_batch = word_inputs[start:start+batch_size]
                char_inputs_batch = char_inputs[start:start+batch_size]
                dict_order_feats_batch = dict_order_feats[start:start+batch_size]
                y_batch = y[start:start+batch_size]
                seqlen_batch = seqlen[start:start+batch_size]
                syllable_model_train_index_batch = np.hstack(syllable_model_train_index[start:start+batch_size])
                syllable_model_train_label_batch = np.hstack(syllable_model_train_label[start:start+batch_size])
                start += batch_size
                _loss_syl_model, _ ,_loss_main, _ = self.sess.run([self.syllable_model_loss, self.syllable_model_optimizer, self.loss, self.optimizer], feed_dict = {self.word_inputs: word_inputs_batch, self.char_inputs: char_inputs_batch, self.dict_order_feats: dict_order_feats_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout, self.is_training: True, self.syllable_model_index: syllable_model_train_index_batch, self.syllable_model_label: syllable_model_train_label_batch})
                # _loss_syl_model, _ = self.sess.run([self.syllable_model_loss, self.syllable_model_optimizer], feed_dict = {self.word_inputs: word_inputs_batch, self.char_inputs: char_inputs_batch, self.dict_order_feats: dict_order_feats_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout, self.is_training: True, self.syllable_model_index: syllable_model_train_index_batch, self.syllable_model_label: syllable_model_train_label_batch})
                sum_loss_main += _loss_main
                sum_loss_syl  += _loss_syl_model
                # print(_loss_syl_model)
            ### evaluate on valid set
            acc_train = self.evaluate(word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train, batch_size)
            acc_test = self.evaluate(word_inputs_valid, char_inputs_valid, dict_order_feats_valid, y_valid, seqlen_valid, batch_size)
            print('epoch: ', epoch, ' loss_main: ', sum_loss_main/(int((len(word_inputs_train)-1)/batch_size) + 1), ' loss_syllable_model: ', sum_loss_syl/(int((len(word_inputs_train)-1)/batch_size) + 1), ' acc_train: ', acc_train, ' acc_test: ', acc_test, ' time: ', str(datetime.now() - start_time))
            f.write('epoch: ' + str(epoch) + ' loss: ' + str(sum_loss_main/(int((len(word_inputs_train)-1)/batch_size) + 1)) + ' loss_syllable_model: ' + str(sum_loss_syl/(int((len(word_inputs_train)-1)/batch_size) + 1)) + " acc_train: " + str(acc_train) + ' acc_test: ' + str(acc_test) + ' time: ' + str(datetime.now() - start_time) + "\n")
            if (acc_test > accuracy):
                accuracy = acc_test
                saver.save(self.sess, save_path + '/model.ckpt')

    def load_trained_model(self, load_path):
        assert os.path.exists(load_path), load_path + ' is not exist!!!'
        saver = tf.train.Saver(self.sess, load_path)

    def save_model(self, save_path):
        saver = tf.train.Saver(max_to_keep = 1000)
        saver.save(self.sess, save_path + '/model.ckpt')

    def load_model(self, load_path):
        saver = tf.train.Saver(max_to_keep = 1000)
        saver.restore(self.sess, load_path)

def load_data(path, inline_path):
    word_data = []
    char_data = []
    labels = []
    seqlens = []
    cnt = 0
    #    sequence model
    with open(path, "r") as f:
        wordvecs = []
        charvecs = []
        labelvecs = []
        for row in f:
            if (row == "\n") and (len(wordvecs) != 0):
                cnt += 1
                print(cnt, end = "\r")
                seqlens.append(min(max_lstm_step, len(wordvecs)))
                while len(wordvecs) < max_lstm_step:
                    charvec = list(np.ones(max_word_length)*char_dictionary.index("<PAD>"))
                    charvecs.append(charvec)
                    wordvecs.append(word_dictionary.index("<PAD>"))
                    labelvecs.append(0)
                word_data.append(wordvecs[:min(max_lstm_step, len(wordvecs))])
                char_data.append(charvecs[:min(max_lstm_step, len(charvecs))])
                labels.append(labelvecs[:min(max_lstm_step, len(wordvecs))])
                wordvecs = []
                labelvecs = []
                charvecs = []
            else:
                word = row[:-1].split("\t")[0]
                label = row[:-1].split("\t")[1]
                if (word in word_dictionary):
                    wordvecs.append(word_dictionary.index(word))
                else:
                    wordvecs.append(word_dictionary.index("<OOV>"))
                charvec = []
                for char in word:
                    if (char in char_dictionary):
                        charvec.append(char_dictionary.index(char))
                    else:
                        charvec.append(char_dictionary.index("<OOV>"))
                while (len(charvec) < max_word_length):
                    charvec.append(char_dictionary.index("<PAD>"))
                charvecs.append(charvec[:min(max_word_length, len(charvec))])
                if (label == "B"):
                    labelvecs.append(1)
                if (label == "I"):
                    labelvecs.append(2)
                if (label == "E"):
                    labelvecs.append(3)
    dict_order_feats = []
    cnt = 0
    print("load dict order...")
    # with open(inline_path, "r") as f:
    #     for row in f:
    #         cnt += 1
    #         print(cnt, end = "\r")
    #         dict_order_feat = []
    #         words = row[:-1].split(" ")
    #         words = ["none", "none"] + words + ["none", "none"]
    #         for i in range(2, len(words) - 2):
    #             feat = [0, 0, 0, 0, 0]
    #             if (' '.join(words[i-2:i+1]) in vi_words) or (' '.join(words[i-2:i+1]).lower() in vi_words):
    #                 feat[0] = 1
    #             if (' '.join(words[i-1:i+1]) in vi_words) or (' '.join(words[i-1:i+1]).lower() in vi_words):
    #                 feat[1] = 1
    #             if (' '.join(words[i-1:i+2]) in vi_words) or (' '.join(words[i-1:i+2]).lower() in vi_words):
    #                 feat[2] = 1
    #             if (' '.join(words[i:i+2]) in vi_words) or (' '.join(words[i:i+2]).lower() in vi_words):
    #                 feat[3] = 1
    #             if (' '.join(words[i:i+3]) in vi_words) or (' '.join(words[i:i+3]).lower() in vi_words):
    #                 feat[4] = 1
    #             dict_order_feat.append(feat)
    #         while (len(dict_order_feat) < max_lstm_step):
    #             dict_order_feat.append([0, 0, 0, 0, 0])
    #         dict_order_feats.append(dict_order_feat[:max_lstm_step])
    #    syllables model
    syllable_model_index = []
    syllable_model_label = []
    for i in range(len(word_data)):
        syllable_index = []
        syllable_label = []
        for j in range(seqlens[i] - 1):
            syllable_index.append(word_data[i][j])
            syllable_label.append(word_data[i][j+1])
        for j in range(seqlens[i] - 2):
            syllable_index.append(word_data[i][j])
            syllable_label.append(word_data[i][j+2])
        syllable_model_index.append(syllable_index)
        syllable_model_label.append(syllable_label)
    return word_data, char_data, labels, seqlens, syllable_model_index, syllable_model_label, dict_order_feats


### parameters
list_labels = ["B", "I", "E"]
word_embedding_size = 100
character_embedding = 100
dict_order_size = 5
#    initial type = [xavier, normal]
initial_type = "normal"
#    parameters for CNN layers
max_word_length = 10
filter_sizes = [1, 2, 3]
num_filters = [30, 30, 40]
#    parameters for LSTM layers
n_lstm_hidden = 200
max_lstm_step = 250
num_lstm_layers = 2
#    parameters for syllable model
num_sampled = 64
#    model parameters
lr = 0.001#01
dropout = 0.8
batch_size = 128
num_epochs = 200

save_path = "models/end_to_end/multi_task+dict_order+cnn+blstm+crf/normal_2_layer_cell_200_we_100_ce_100_lr_0.001_dropout_0.8_cnn_1_30_2_30_3_40"

### load word dictionary
word_dictionary = []
with open("data/word_dictionary.txt", "r") as f:
    for row in f:
        word_dictionary.append(row[:-1])
print("word dictionary length: ", len(word_dictionary))

### load character dictionary
char_dictionary = []
with open("data/char_dictionary.txt", "r") as f:
    for row in f:
        char_dictionary.append(row[:-1])
print("character dictionary length: ", len(char_dictionary))

### load vietnamese word list
vi_words = []
with open("data/full_vocab.txt", "r") as f:
    for row in f:
        vi_words.append(row[:-1])

### load data
print("load train data...")
test_word_data, test_char_data, test_label, test_seqlen, syllable_model_test_index, syllable_model_test_label, test_dict_order_feats = load_data("data/VTB-train-dev-test/test-BIE", "data/VTB-train-dev-test/test-inline")
with open("data/VTB-train-dev-test/test_dict_order.pkl", "rb") as f:
    test_dict_order_feats = pickle.load(f)
# with open("data/VTB-train-dev-test/test_dict_order.pkl", "wb") as f:
#     pickle.dump(test_dict_order_feats, f)
print(np.shape(test_word_data))
print(np.shape(test_char_data))
print(np.shape(test_dict_order_feats))
print(np.shape(test_label))
print(np.shape(test_seqlen))
print(np.shape(syllable_model_test_index))
print(np.shape(syllable_model_test_label))

### train new model
print("train new model")
segmenter = DeepCNNBLSTMSegment(n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
                                char_dictionary, character_embedding, max_word_length, filter_sizes, \
                                num_filters, initial_type, dict_order_size, num_sampled)
segmenter.load_model(save_path + "/model.ckpt")
print(segmenter.evaluate_f1(test_word_data, test_char_data, test_dict_order_feats, test_label, test_seqlen, batch_size))
