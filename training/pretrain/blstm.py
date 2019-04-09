import numpy as np
import tensorflow as tf
from random import shuffle
from datetime import datetime
import fasttext as ft
import pickle
import os

class DeepBLSTMSegment():
    def __init__(self, n_hidden, max_step, num_layers):
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
        # self.sess = tf.Session(config = config)
        self.sess = tf.InteractiveSession()
        self.n_hidden = n_hidden
        self.max_step = max_step
        self.num_layers = num_layers
        self.num_classes = 2
        self.build_model()

    def build_model(self):
        self.init_placeholder()
        self.build_graph()
        self.loss_optimizer()
        self.sess.run(tf.global_variables_initializer())
        # summary_writer = tf.train.SummaryWriter('/home/tittit/python/web_mining2/logs', graph = tf.get_default_graph())

    def init_placeholder(self):
        ### placeholders
        self.x = tf.placeholder(tf.float64, [None, self.max_step, 100])
        self.y = tf.placeholder(tf.float64, [None, self.max_step, self.num_classes])
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.lr = tf.placeholder(tf.float64, None)
        self.dropout = tf.placeholder(tf.float64, None)
        self.current_batch_size = tf.shape(self.x)[0]

    def build_graph(self):
        ### softmax params
        self.w = tf.Variable(tf.truncated_normal([2*self.n_hidden, self.num_classes], dtype = tf.float64), name = 'w', dtype = tf.float64)
        self.b = tf.Variable(tf.truncated_normal([1, self.num_classes], dtype = tf.float64), name = 'b', dtype = tf.float64)
        ### LSTM layer
        self.output = self.x
        self.lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout) for i in range(self.num_layers)]
        self.lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout) for i in range(self.num_layers)]
        for i in range(self.num_layers):
            (self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.lstm_cells_fw[i], cell_bw = self.lstm_cells_bw[i], inputs = self.output, sequence_length = self.sequence_length, scope = 'B_LSTM_layer_' + str(i), dtype = tf.float64)
            self.output = tf.concat([self.output_fw, self.output_bw], axis = -1)
        self.output_slice = tf.reshape(tf.nn.relu(self.output), [-1, 2*self.n_hidden])
        self.pred_slice = tf.nn.softmax(tf.matmul(self.output_slice, self.w) + self.b)
        self.pred = tf.reshape(self.pred_slice, [-1, self.max_step, self.num_classes])

    def loss_optimizer(self):
        self.loss = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(self.y * tf.log(self.pred), axis = 2), axis = 1), axis = 0)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def shuffle(self, x, y, seqlen):
        z = list(zip(x, y, seqlen))
        shuffle(z)
        x, y, seqlen = zip(*z)
        return x, y, seqlen

    def evaluate(self, x_test, y_test, seqlen_test, batch_size):
        start = 0
        cnt = 0
        sum_cnt = np.sum(seqlen_test)
        while (start < len(seqlen_test)):
            batch_data = x_test[start:start + batch_size]
            batch_labels = y_test[start:start + batch_size]
            batch_seqlen = seqlen_test[start:start + batch_size]
            start += batch_size
            _pred = self.sess.run(self.pred, feed_dict = {self.x: batch_data, self.sequence_length: batch_seqlen, self.dropout: 1.0})
            for i in range(len(batch_seqlen)):
                cnt += np.sum(np.equal(np.argmax(_pred[i][:batch_seqlen[i]], axis = 1), np.argmax(batch_labels[i][:batch_seqlen[i]], axis = 1)))
        return cnt/sum_cnt

    def train_new_model(self, x_train, y_train, seqlen_train, x_valid, y_valid, seqlen_valid, lr, dropout, batch_size, num_epochs, save_path):
        accuracy = 0
        saver = tf.train.Saver(max_to_keep = 1000)
        f = open(os.path.join(save_path, "log.txt"), "w")
        for epoch in range(num_epochs):
            ### shuffle all training data
            x, y, seqlen = self.shuffle(x_train, y_train, seqlen_train)
            sum_loss = 0
            start_time = datetime.now()
            while (len(x) != 0):
                ### get training batch
                upperbound = min(batch_size, len(x))
                x_batch = x[:upperbound]
                y_batch = y[:upperbound]
                seqlen_batch = seqlen[:upperbound]
                x = x[upperbound:]
                y = y[upperbound:]
                seqlen = seqlen[upperbound:]
                _loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout})
                sum_loss += _loss
            ### evaluate on valid set
            f1_score = self.evaluate(x_valid, y_valid, seqlen_valid, batch_size)
            print('epoch: ', epoch, ' loss: ', sum_loss/(int((len(x_train)-1)/batch_size) + 1), ' f1_score: ', f1_score, ' time: ', str(datetime.now() - start_time))
            f.write('epoch: ' + str(epoch) + ' loss: ' + str(sum_loss/(int((len(x_train)-1)/batch_size) + 1)) + ' f1_score: ' + str(f1_score) + ' time: ' + str(datetime.now() - start_time) + "\n")
            if (f1_score > accuracy):
                accuracy = f1_score
                saver.save(self.sess, save_path + '/model.ckpt')

    def train_new_partial_model(self, x_train, y_train, seqlen_train, lr, batch_size, dropout):
        x, y, seqlen = self.shuffle(x_train, y_train, seqlen_train)
        sum_loss = 0
        while (len(x) != 0):
            ### get training batch
            upperbound = min(batch_size, len(x))
            x_batch = x[:upperbound]
            y_batch = y[:upperbound]
            seqlen_batch = seqlen[:upperbound]
            x = x[upperbound:]
            y = y[upperbound:]
            seqlen = seqlen[upperbound:]
            _loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout})
            sum_loss += _loss
        return sum_loss

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
    data = []
    labels = []
    seqlens = []
    cnt = 0
    with open(path, "r") as f:
        wordvecs = []
        labelvecs = []
        for row in f:
            if (row == "\n") and (len(wordvecs) != 0):
                cnt += 1
                print(cnt, end = "\r")
                seqlens.append(min(max_step, len(wordvecs)))
                while len(wordvecs) < max_step:
                    wordvecs.append(np.zeros(100))
                    labelvecs.append(np.zeros(2))
                data.append(wordvecs[:min(max_step, len(wordvecs))])
                labels.append(labelvecs[:min(max_step, len(wordvecs))])
                wordvecs = []
                labelvecs = []
            else:
                word = row[:-1].split("\t")[0]
                label = row[:-1].split("\t")[1]
                wordvecs.append(word2vec[word])
                labelvec = np.zeros(2)
                labelvec[list_labels.index(label)] = 1
                labelvecs.append(labelvec)
    return data, labels, seqlens

### parameters
list_labels = ["B", "I"]
n_hidden = 200
max_step = 250
num_layers = 2
lr = 0.01
dropout = 0.5
batch_size = 128
num_epochs = 200
save_path = "models/blstm/2_layers_biLSTM_maxstep_250_cell_200_dropoutoutput_0.5_lr_0.01"

### load pretrain models
word2vec = ft.load_model("models/word2vec/model.bin")

### load data
print("load train data...")
train_data, train_label, train_seqlen = load_data("data/VTB-train-dev-test/train-BI")
print(np.shape(train_data))
print(np.shape(train_label))
print(np.shape(train_seqlen))

print("load dev data...")
dev_data, dev_label, dev_seqlen = load_data("data/VTB-train-dev-test/dev-BI")
print(np.shape(dev_data))
print(np.shape(dev_label))
print(np.shape(dev_seqlen))

print("load test data...")
test_data, test_label, test_seqlen = load_data("data/VTB-train-dev-test/test-BI")
print(np.shape(test_data))
print(np.shape(test_label))
print(np.shape(test_seqlen))

### train new model
print("train new model")
segmenter = DeepBLSTMSegment(n_hidden, max_step, num_layers)
segmenter.train_new_model(train_data, train_label, train_seqlen, dev_data, dev_label, dev_seqlen, lr, dropout, batch_size, num_epochs, save_path)
