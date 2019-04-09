import numpy as np
import tensorflow as tf
from random import shuffle
from datetime import datetime
from tensorflow.python.layers.core import Dense
from utils.lstm import BNLSTMCell
import pickle
import math
import os

class DeepCNNBLSTMSegment():
	def __init__(self, n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
				character_dictionary, character_embedding_size, max_word_length, filter_sizes, num_filters, \
				initial_type):
		#	initialize session
		self.sess = tf.InteractiveSession()
		#	parameters for inputs
		self.word_dictionary = word_dictionary
		self.word_embedding_size = word_embedding_size
		self.character_dictionary = character_dictionary
		self.character_embedding_size = character_embedding_size
		#	initial type = [xavier, normal]
		self.initial_type = initial_type
		#	parameters for CNN layers
		self.max_word_length = max_word_length
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		#	parameters for LSTM layers
		self.n_lstm_hidden = n_lstm_hidden
		self.max_lstm_step = max_lstm_step
		self.num_lstm_layers = num_lstm_layers
		#	parameters for outputs
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
		self.word_inputs = tf.placeholder(tf.int32, [None, self.max_lstm_step])
		self.char_inputs = tf.placeholder(tf.int32, [None, self.max_lstm_step, self.max_word_length])
		self.y = tf.placeholder(tf.float64, [None, self.max_lstm_step, self.num_classes])
		self.sequence_length = tf.placeholder(tf.int32, [None])
		self.lr = tf.placeholder(tf.float64, None)
		self.dropout = tf.placeholder(tf.float64, None)
		self.is_training = tf.placeholder(tf.bool, None)
		self.current_batch_size = tf.shape(self.word_inputs)[0]

	def build_graph(self):
		#	define type of initial variables
		if self.initial_type == "normal":
			#	initialize encoder embedding have variance = 1
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype = tf.float64)
		elif self.initial_type == "xavier":
			initializer = tf.contrib.layers.xavier_initializer()

		### word embedding layer
		with tf.name_scope('word_embedding_layer'):
			word_embedding_matrix = tf.get_variable(name = 'word_embedding_matrix', shape = [len(self.word_dictionary), self.word_embedding_size], initializer = initializer, dtype = tf.float64)
			# word inputs embedding: [batch size, max_lstm_step, embedding size]
			word_inputs_embedding = tf.nn.embedding_lookup(params = word_embedding_matrix, ids = self.word_inputs)

		### character embedding layer
		with tf.name_scope('character_embedding_layer'):
			character_embedding_matrix = tf.get_variable(name = 'character_embedding_matrix', shape = [len(self.character_dictionary), self.character_embedding_size], initializer = initializer, dtype = tf.float64)
			character_inputs_embedding = tf.nn.embedding_lookup(params = character_embedding_matrix, ids = self.char_inputs)
			# character inputs embedding: [batch size, max_lstm_step, max_word_length, embedding size]
			# print(self.char_inputs)
			# char_inputs_slice = tf.reshape(self.char_inputs, [-1, self.max_word_length])
			# print(char_inputs_slice)
			# raw_character_inputs_embedding = tf.nn.embedding_lookup(params = character_embedding_matrix, ids = char_inputs_slice)
			# print(raw_character_inputs_embedding)
			# character_inputs_embedding = tf.reshape(raw_character_inputs_embedding, [self.current_batch_size, self.max_lstm_step, self.max_word_length, self.character_embedding_size])
			# print(character_inputs_embedding)
		
		### CNN layer
		with tf.name_scope("CNN_layer"):
			cnn_feats = []
			#	reshape character input embedding to [batch size*max_lstm_step, max_word_length, embedding size, 1]
			images = tf.reshape(character_inputs_embedding, [self.current_batch_size*self.max_lstm_step, self.max_word_length, self.character_embedding_size, 1])
			for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters)):
				w = tf.get_variable(shape = [filter_size, self.character_embedding_size, 1, num_filter], name = "w_conv_" + str(i), initializer = initializer, dtype = tf.float64)
				b = tf.get_variable(shape = [num_filter], name = "b_conv_" + str(i), initializer = initializer, dtype = tf.float64)
				#	conv has shape = [batchsize*max_lstm_step, k, 1, num_filter]
				conv = tf.nn.relu(tf.nn.conv2d(images, w, strides = [1, 1, 1, 1], padding = 'VALID'))
				#	conv_pool has shape [batchsize*max_lstm_step, 1, 1, num_filter]
				conv_pool = tf.reduce_max(conv, axis = 1)
				#	conv_feat has shape [batsize, max_lstm_step, num_filter]
				conv_feat = tf.reshape(conv_pool, [self.current_batch_size, self.max_lstm_step, num_filter])
				cnn_feats.append(conv_feat)
			output_cnn = cnn_feats[0]
			for cnn_feat in cnn_feats[1:]:
				output_cnn = tf.concat([output_cnn, cnn_feat], axis = -1)

		### LSTM layer
		with tf.name_scope("biLSTM_layer"):
			output = tf.concat([word_inputs_embedding, output_cnn], axis = -1)
			lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_lstm_hidden), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
			lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_lstm_hidden), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
			# lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(BNLSTMCell(num_units = self.n_lstm_hidden, is_training_tensor = self.is_training, max_bn_steps = self.max_lstm_step, dtype = tf.float64), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
			# lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(BNLSTMCell(num_units = self.n_lstm_hidden, is_training_tensor = self.is_training, max_bn_steps = self.max_lstm_step, dtype = tf.float64), output_keep_prob = self.dropout) for i in range(self.num_lstm_layers)]
			for i in range(self.num_lstm_layers):
				(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cells_fw[i], cell_bw = lstm_cells_bw[i], inputs = output, sequence_length = self.sequence_length, scope = 'B_LSTM_layer_' + str(i), dtype = tf.float64)
				output = tf.concat([output_fw, output_bw], axis = -1)
		
		with tf.name_scope("softmax_layer"):
			w = tf.get_variable(shape = [2*self.n_lstm_hidden, self.num_classes], name = 'w', initializer = initializer, dtype = tf.float64)
			b = tf.get_variable(shape = [1, self.num_classes], name = 'b', initializer = initializer, dtype = tf.float64)
			output_slice = tf.reshape(tf.nn.relu(output), [-1, 2*self.n_lstm_hidden])
			pred_slice = tf.nn.softmax(tf.matmul(output_slice, w) + b)
			self.pred = tf.reshape(pred_slice, [-1, self.max_lstm_step, self.num_classes])

	def loss_optimizer(self):
		self.loss = tf.reduce_mean(tf.reduce_sum(-tf.reduce_sum(self.y * tf.log(self.pred), axis = 2), axis = 1), axis = 0)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def shuffle(self, word_inputs, char_inputs, y, seqlen):
		z = list(zip(word_inputs, char_inputs, y, seqlen))
		shuffle(z)
		word_inputs, char_inputs, y, seqlen = zip(*z)
		return word_inputs, char_inputs, y, seqlen

	def evaluate(self, word_inputs_test, char_inputs_test, y_test, seqlen_test, batch_size):
		start = 0
		cnt = 0
		sum_cnt = np.sum(seqlen_test)
		while (start < len(seqlen_test)):
			batch_word_inputs_data = word_inputs_test[start:start + batch_size]
			batch_char_inputs_data = char_inputs_test[start:start + batch_size]
			batch_labels = y_test[start:start + batch_size]
			batch_seqlen = seqlen_test[start:start + batch_size]
			start += batch_size
			_pred = self.sess.run(self.pred, feed_dict = {self.word_inputs: batch_word_inputs_data, self.char_inputs: batch_char_inputs_data, self.sequence_length: batch_seqlen, self.dropout: 1.0, self.is_training: False})
			for i in range(len(batch_seqlen)):
				cnt += np.sum(np.equal(np.argmax(_pred[i][:batch_seqlen[i]], axis = 1), np.argmax(batch_labels[i][:batch_seqlen[i]], axis = 1)))
		return cnt/sum_cnt

	def train_new_model(self, word_inputs_train, char_inputs_train, y_train, seqlen_train, word_inputs_valid, char_inputs_valid, y_valid, seqlen_valid, lr, dropout, batch_size, num_epochs, save_path):
		accuracy = 0
		saver = tf.train.Saver(max_to_keep = 1000)
		f = open(os.path.join(save_path, "log.txt"), "a")
		for epoch in range(num_epochs):
			### shuffle all training data
			word_inputs, char_inputs, y, seqlen = self.shuffle(word_inputs_train, char_inputs_train, y_train, seqlen_train)
			sum_loss = 0
			start_time = datetime.now()
			while (len(word_inputs) != 0):
				### get training batch
				word_inputs_batch = word_inputs[:batch_size]
				char_inputs_batch = char_inputs[:batch_size]
				y_batch = y[:batch_size]
				seqlen_batch = seqlen[:batch_size]
				word_inputs = word_inputs[batch_size:]
				char_inputs = char_inputs[batch_size:]
				y = y[batch_size:]
				seqlen = seqlen[batch_size:]
				_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.word_inputs: word_inputs_batch, self.char_inputs: char_inputs_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout, self.is_training: True})
				sum_loss += _loss
			### evaluate on valid set
			acc_train = self.evaluate(word_inputs_train, char_inputs_train, y_train, seqlen_train, batch_size)
			acc_test = self.evaluate(word_inputs_valid, char_inputs_valid, y_valid, seqlen_valid, batch_size)
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
					labelvecs.append(np.zeros(2))
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
					wordvecs.append(word_dictionary.index(word.lower()))
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
				labelvec = np.zeros(2)
				labelvec[list_labels.index(label)] = 1
				labelvecs.append(labelvec)
	return word_data, char_data, labels, seqlens


### parameters
list_labels = ["B", "I"]
word_embedding_size = 50
character_embedding = 50
#	initial type = [xavier, normal]
initial_type = "normal"
#	parameters for CNN layers
max_word_length = 10
filter_sizes = [1, 2, 3]
num_filters = [30, 30, 40]
#	parameters for LSTM layers
n_lstm_hidden = 200
max_lstm_step = 250
num_lstm_layers = 2
#	model parameters
lr = 0.001
dropout = 0.8
batch_size = 128
num_epochs = 200

save_path = "models/embed_cnn_blstm/normal_2_layer_cell_200_we_50_ce_50_lr_0.01_dropout_0.8_cnn_1_30_2_30_3_40_word_lower_form"

### load word dictionary
word_dictionary = []
with open("data/lower_word_dictionary.txt", "r") as f:
	for row in f:
		word_dictionary.append(row[:-1])
print("word dictionary length: ", len(word_dictionary))

### load character dictionary
char_dictionary = []
with open("data/char_dictionary.txt", "r") as f:
	for row in f:
		char_dictionary.append(row[:-1])
print("character dictionary length: ", len(char_dictionary))

### load data
print("load train data...")
train_word_data, train_char_data, train_label, train_seqlen = load_data("data/VTB-train-dev-test/train-BI")
print(np.shape(train_word_data))
print(np.shape(train_char_data))
print(np.shape(train_label))
print(np.shape(train_seqlen))

print("load dev data...")
dev_word_data, dev_char_data, dev_label, dev_seqlen = load_data("data/VTB-train-dev-test/dev-BI")
print(np.shape(dev_word_data))
print(np.shape(dev_char_data))
print(np.shape(dev_label))
print(np.shape(dev_seqlen))

### train new model
print("train new model")
segmenter = DeepCNNBLSTMSegment(n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
								char_dictionary, character_embedding, max_word_length, filter_sizes, \
								num_filters, initial_type)
# segmenter.load_model("models/embed_cnn_blstm/normal_2_layer_cell_200_lr_0.01_dropout_0.8_cnn_2_20_3_20_4_20/model.ckpt")
segmenter.train_new_model(train_word_data, train_char_data, train_label, train_seqlen, dev_word_data, dev_char_data, dev_label, dev_seqlen, lr, dropout, batch_size, num_epochs, save_path)
