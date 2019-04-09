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

class LM(object):
    def __init__(self, embed):
        self.embed = embed

    def get(self, w):
        if (w in self.embed.keys()):
            return self.embed[w]
        return self.embed["<OOV>"]

class DeepCNNBLSTMSegment():
    def __init__(self, n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
                character_dictionary, character_embedding_size, max_word_length, filter_sizes, num_filters, \
                initial_type, dict_order_size, word_embedding_initializer):
        #    initialize session
        self.sess = tf.InteractiveSession()
        #    parameters for inputs
        self.word_dictionary = word_dictionary
        self.word_embedding_size = word_embedding_size
        self.word_embedding_initializer = word_embedding_initializer
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
            self.word_embedding_matrix = tf.get_variable(name = 'word_embedding_matrix', initializer = self.word_embedding_initializer, dtype = tf.float32)
            # word inputs embedding: [batch size, max_lstm_step, embedding size]
            word_inputs_embedding = tf.nn.embedding_lookup(params = self.word_embedding_matrix, ids = self.word_inputs)

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
                precision_cnt += batch_seqlen[i] - np.sum(np.argmax(_pred[i], axis = -1)[:batch_seqlen[i]])
                recall_cnt += batch_seqlen[i] - np.sum(np.argmax(batch_labels[i], axis = -1)[:batch_seqlen[i]])
                s = 0
                e = 0
                true_labels = np.argmax(batch_labels[i], axis = -1)
                pred_labels = np.argmax(_pred[i], axis = -1)
                while (e < batch_seqlen[i]):
                    e += 1
                    if (e == batch_seqlen[i]) or (true_labels[e] == 0):
                        if (list(true_labels[s:e]) == list(pred_labels[s:e])):
                            cnt += 1
                        s = e
        precision = cnt/precision_cnt
        recall = cnt/recall_cnt
        f1 = 2*precision*recall/(precision + recall)
        return precision, recall, f1

    def train_new_model(self, word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train, word_inputs_valid, char_inputs_valid, dict_order_feats_valid, y_valid, seqlen_valid, lr, dropout, batch_size, num_epochs, save_path):
        accuracy = 0
        saver = tf.train.Saver(max_to_keep = 1000)
        f = open(os.path.join(save_path, "log.txt"), "w")
        for epoch in range(num_epochs):
            ### shuffle all training data
            word_inputs, char_inputs, dict_order_feats, y, seqlen = self.shuffle(word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train)
            sum_loss = 0
            start_time = datetime.now()
            while (len(word_inputs) != 0):
                ### get training batch
                word_inputs_batch = word_inputs[:batch_size]
                char_inputs_batch = char_inputs[:batch_size]
                dict_order_feats_batch = dict_order_feats[:batch_size]
                y_batch = y[:batch_size]
                seqlen_batch = seqlen[:batch_size]
                word_inputs = word_inputs[batch_size:]
                char_inputs = char_inputs[batch_size:]
                dict_order_feats = dict_order_feats[batch_size:]
                y = y[batch_size:]
                seqlen = seqlen[batch_size:]
                _loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.word_inputs: word_inputs_batch, self.char_inputs: char_inputs_batch, self.dict_order_feats: dict_order_feats_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout, self.is_training: True})
                sum_loss += _loss
            ### evaluate on valid set
            acc_train = self.evaluate(word_inputs_train, char_inputs_train, dict_order_feats_train, y_train, seqlen_train, batch_size)
            acc_test = self.evaluate(word_inputs_valid, char_inputs_valid, dict_order_feats_valid, y_valid, seqlen_valid, batch_size)
            print('epoch: ', epoch, ' loss: ', sum_loss/(int((len(word_inputs_train)-1)/batch_size) + 1), ' acc_train: ', acc_train, ' acc_test: ', acc_test, ' time: ', str(datetime.now() - start_time))
            f.write('epoch: ' + str(epoch) + ' loss: ' + str(sum_loss/(int((len(word_inputs_train)-1)/batch_size) + 1)) + " acc_train: " + str(acc_train) + ' acc_test: ' + str(acc_test) + ' time: ' + str(datetime.now() - start_time) + "\n")
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

def load_data(path):
    word_data = []
    char_data = []
    labels = []
    seqlens = []
    cnt = 0
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
    return word_data, char_data, labels, seqlens


### parameters
list_labels = ["B", "I", "E"]
word_embedding_size = 100
character_embedding = 100
dict_order_size = 5
#    initial type = [xavier, normal]
initial_type = "normal"
word_vector_model = "lm"
#    parameters for CNN layers
max_word_length = 10
filter_sizes = [1, 2, 3]
num_filters = [30, 30, 40]
#    parameters for LSTM layers
n_lstm_hidden = 200
max_lstm_step = 250
num_lstm_layers = 2
#    model parameters
lr = 0.001
dropout = 0.8
batch_size = 128
num_epochs = 100

save_path = "models/pretrain/wv_init_dict_order_cnn_blstm_crf/lm/normal_2_layer_cell_200_we_100_ce_100_lr_0.001_dropout_0.8_cnn_1_30_2_30_3_40"

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

### load word embedding initializer
word_embedding_initializer = []
if word_vector_model == "ft":
    word2vec = ft.load_model("models/word2vec/model.bin")
    for word in word_dictionary:
        word_embedding_initializer.append(list(word2vec[word]))
if word_vector_model == "lm":
    with open("models/lm/embedding/model.pkl", "rb") as f:
        word2vec = pickle.load(f)
    for word in word_dictionary:
        word_embedding_initializer.append(list(word2vec.get(word)))
print(np.shape(word_embedding_initializer))

### load data
print("load train data...")
train_word_data, train_char_data, train_label, train_seqlen = load_data("data/VTB-train-dev-test/train-BIE")
with open("data/VTB-train-dev-test/train_dict_order.pkl", "rb") as f:
    train_dict_order_feats = pickle.load(f)
print(np.shape(train_word_data))
print(np.shape(train_char_data))
print(np.shape(train_dict_order_feats))
print(np.shape(train_label))
print(np.shape(train_seqlen))

print("load dev data...")
dev_word_data, dev_char_data, dev_label, dev_seqlen = load_data("data/VTB-train-dev-test/dev-BIE")
with open("data/VTB-train-dev-test/dev_dict_order.pkl", "rb") as f:
    dev_dict_order_feats = pickle.load(f)
print(np.shape(dev_word_data))
print(np.shape(dev_char_data))
print(np.shape(dev_dict_order_feats))
print(np.shape(dev_label))
print(np.shape(dev_seqlen))

### train new model
print("train new model")
segmenter = DeepCNNBLSTMSegment(n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
                                char_dictionary, character_embedding, max_word_length, filter_sizes, \
                                num_filters, initial_type, dict_order_size, word_embedding_initializer)
# segmenter.load_model("models/embed_cnn_blstm/normal_2_layer_cell_200_lr_0.01_dropout_0.8_cnn_2_20_3_20_4_20/model.ckpt")
segmenter.train_new_model(train_word_data, train_char_data, train_dict_order_feats, train_label, train_seqlen, dev_word_data, dev_char_data, dev_dict_order_feats, dev_label, dev_seqlen, lr, dropout, batch_size, num_epochs, save_path)

