import numpy as np
import tensorflow as tf
from random import shuffle
from datetime import datetime
from tensorflow.python.layers.core import Dense
import pickle
import math
import os

class DeepBLSTMSegment():
	def __init__(self, n_hidden, max_step, num_layers, dictionary, embedding_size):
		# config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
		# self.sess = tf.Session(config = config)
		self.sess = tf.InteractiveSession()
		self.dictionary = dictionary
		self.embedding_size = embedding_size
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
		self.x = tf.placeholder(tf.int32, [None, self.max_step])
		self.y = tf.placeholder(tf.float32, [None, self.max_step, self.num_classes])
		self.sequence_length = tf.placeholder(tf.int32, [None])
		self.lr = tf.placeholder(tf.float32, None)
		self.dropout = tf.placeholder(tf.float32, None)
		self.current_batch_size = tf.shape(self.x)[0]

	def build_graph(self):
		with tf.name_scope('embedding_layer'):
			# initialize encoder embedding have variance = 1
			sqrt3 = math.sqrt(3)
			initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype = tf.float32)
			self.encoder_embedding = tf.get_variable(name = 'embedding', shape = [len(self.dictionary), self.embedding_size], initializer = initializer, dtype = tf.float32)
			# inputs embedding: [batch size, max_step, embedding size]
			self.inputs_embedded = tf.nn.embedding_lookup(params = self.encoder_embedding, ids = self.x)
			# input layer
			input_layer = Dense(self.n_hidden, dtype = tf.float32, name = 'input_projection')
			self.inputs_embedded = input_layer(self.inputs_embedded)
		
		### LSTM layer
		with tf.name_scope("biLSTM_layer"):
			self.output = self.inputs_embedded
			self.lstm_cells_fw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout) for i in range(self.num_layers)]
			self.lstm_cells_bw = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout) for i in range(self.num_layers)]
			for i in range(self.num_layers):
				(self.output_fw, self.output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.lstm_cells_fw[i], cell_bw = self.lstm_cells_bw[i], inputs = self.output, sequence_length = self.sequence_length, scope = 'B_LSTM_layer_' + str(i), dtype = tf.float32)
				self.output = tf.concat([self.output_fw, self.output_bw], axis = -1)
		
		with tf.name_scope("softmax_layer"):
			self.w = tf.Variable(tf.truncated_normal([2*self.n_hidden, self.num_classes], dtype = tf.float32), name = 'w', dtype = tf.float32)
			self.b = tf.Variable(tf.truncated_normal([1, self.num_classes], dtype = tf.float32), name = 'b', dtype = tf.float32)
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

	def evaluate_f1(self, x_test, y_test, seqlen_test, batch_size):
		start = 0
		cnt = 0
		recall_cnt = 0
		precision_cnt = 0
		while (start < len(seqlen_test)):
			batch_data = x_test[start:start + batch_size]
			batch_labels = y_test[start:start + batch_size]
			batch_seqlen = seqlen_test[start:start + batch_size]
			start += batch_size
			_pred = self.sess.run(self.pred, feed_dict = {self.x: batch_data, self.sequence_length: batch_seqlen, self.dropout: 1.0})
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
					wordvecs.append(dictionary.index("<OOV>"))
					labelvecs.append(np.zeros(2))
				data.append(wordvecs[:min(max_step, len(wordvecs))])
				labels.append(labelvecs[:min(max_step, len(wordvecs))])
				wordvecs = []
				labelvecs = []
			else:
				word = row[:-1].split("\t")[0]
				label = row[:-1].split("\t")[1]
				if (word in dictionary):
					wordvecs.append(dictionary.index(word))
				else:
					wordvecs.append(dictionary.index("<OOV>"))
				labelvec = np.zeros(2)
				labelvec[list_labels.index(label)] = 1
				labelvecs.append(labelvec)
	return data, labels, seqlens



### parameters
list_labels = ["B", "I"]
embedding_size = 100
n_hidden = 200
max_step = 250
num_layers = 2
lr = 0.01
dropout = 0.8
batch_size = 128
num_epochs = 200
save_path = "models/embed_blstm/"

### load dictionary
dictionary = []
with open("data/word_dictionary.txt", "r") as f:
	for row in f:
		dictionary.append(row[:-1])
print("dictionary length: ", len(dictionary))

### load data
print("load test data...")
test_data, test_label, test_seqlen = load_data("data/VTB-train-dev-test/test-BI")
print(np.shape(test_data))
print(np.shape(test_label))
print(np.shape(test_seqlen))

### train new model
print("train new model")
segmenter = DeepBLSTMSegment(n_hidden, max_step, num_layers, dictionary, embedding_size)
segmenter.load_model("models/end_to_end/embed_blstm/2_layer_cell_200/model.ckpt")
print(segmenter.evaluate_f1(test_data, test_label, test_seqlen, batch_size))

#######
sentences = []
sentence = ""
with open("data/VTB-train-dev-test/test-BI", "r") as f:
	for row in f:
		if row == "\n" :
			sentences.append(sentence[:-1])
			sentence = ""
		else:
			sentence += row[:-1].split("\t")[0] + " "

cnt = 0
with open("res.txt", "w") as f:
	for sentence, word_data, char_data, seqlen in zip(sentences, test_word_data, test_char_data, test_seqlen):
		cnt += 1
		print(cnt, end = "\r")
		label = segmenter.predict(word_data, char_data, seqlen)
		words = sentence.split(" ")
		sentence_tokenize = ""
		for i in range(len(words)):
			if (label[i] == 1):
				sentence_tokenize += "_"
			else:
				sentence_tokenize += " "
			sentence_tokenize += words[i]
		# print(sentence_tokenize)
		# probas = segmenter.predict_proba(word_data, char_data, seqlen)
		# for word, proba in zip(words, probas):
		# 	print(word, " ", proba)
		f.write(sentence_tokenize + "\n")

#######