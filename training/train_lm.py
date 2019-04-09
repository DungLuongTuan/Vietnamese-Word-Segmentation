import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import math
import pdb
import os

data_path = "/home/common_gpu1/corpora/nlp/lm_corpus/thanhlc/line_corpus/full.txt"
#    make dictionary
print("make dictionary")
if os.path.exists("data/lm/dictionary.txt"):
    dictionary = []
    with open("data/lm/dictionary.txt", "r") as f:
        for row in f:
            dictionary.append(row[:-1])
else:
    dictionary = set()
    with open(data_path, "r") as f:
        for row in f:
            words = row[:-1].split(" ")
            for word in words:
                dictionary.add(word)
    dictionary = list(dictionary)
    dictionary.sort()
    with open("data/lm/dictionary.txt", "w") as f:
        for word in dictionary:
            f.write(word + "\n")

#    make dictionary frequency
print("make dictionary frequency")
if os.path.exists("data/lm/dictionary_freq.txt"):
    dictionary_freq = {}
    with open("data/lm/dictionary_freq.txt", "r") as f:
        for row in f:
            split_row = row[:-1].split("\t")
            dictionary_freq[split_row[0]] = int(split_row[1])
else:
    dictionary_freq = {}
    for word in dictionary:
        dictionary_freq[word] = 0
    cnt = 0
    with open(data_path, "r") as f:
        for row in f:
            cnt += 1
            print(cnt, end = "\r")
            words = row[:-1].split(" ")
            for word in words:
                dictionary_freq[word] += 1
    with open("data/lm/dictionary_freq.txt", "w") as f:
        for word in dictionary:
            f.write(word + "\t" + str(dictionary_freq[word]) + "\n")

#    parameters
min_freq = 10
word_n_gram = 3
embedding_size = 100
num_sampled = 64
epochs = 10
batch_size = 128

#    load dictionary
print("load dictionary")
word_dictionary = []
cnt = 0
for word in dictionary:
    cnt += 1
    print(cnt, end = "\r")
    if dictionary_freq[word] > min_freq:
        word_dictionary.append(word)
word_dictionary.sort()
word_dictionary.append("<OOV>")
word_index = {}
for i, word in enumerate(word_dictionary):
    word_index[word] = i
print(len(word_dictionary))
### train lm
#   load data
print("load data...")
if (not os.path.exists("data/lm/train_data.txt")):
    cnt = 0
    f_inp = open(data_path, "r")
    f_out = open("data/lm/train_data.txt", "w")
    for row in f_inp:
        cnt += 1
        print(cnt, end = "\r")
        words = row[:-1].split(" ")
        for i, wordi in enumerate(words):
            for wordj in words[i+1:i+word_n_gram]:
                if wordi in word_index.keys():
                    idxi = word_index[wordi]
                else:
                    idxi = word_index["<OOV>"]
                if wordj in word_index.keys():
                    idxj = word_index[wordj]
                else:
                    idxj = word_index["<OOV>"]
                f_out.write(str(idxi) + "\t" + str(idxj) + "\n")
    f_inp.close()
    f_out.close()

#   train new language model
embeddings = tf.Variable(tf.random_uniform([len(word_dictionary), embedding_size], -1.0, 1.0))
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm

nce_weights = tf.Variable(tf.truncated_normal([len(word_dictionary), embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([len(word_dictionary)]))

train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[None, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=len(word_dictionary)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
for epoch in range(epochs):
    sum_loss = 0
    with open("data/lm/train_data.txt", "r") as f:
        cnt_batch = 0
        inp = []
        out = []
        cnt = 0
        for row in f:
            cnt += 1
            print(cnt, end = "\r")
            inp.append(int(row[:-1].split("\t")[0]))
            out.append([int(row[:-1].split("\t")[1])])
            cnt_batch += 1
            if (cnt_batch % batch_size) == 0:
                pdb.set_trace()
                _, _loss = sess.run([optimizer, loss], feed_dict = {train_inputs: inp, train_labels: out})
                sum_loss += _loss/batch_size
                inp = []
                out = []
                cnt_batch = 0
    print("epoch: ", epoch, "\tloss:  ", sum_loss)
#     saver.save(sess, "models/lm/model.ckpt")

# #   load pretrain language model
# saver.restore(sess, "models/lm/model.ckpt")
# text1 = 'con'
# text2 = 'mẹ'
# a = sess.run(embed, feed_dict = {train_inputs: [word_index[text1]]})
# b = sess.run(embed, feed_dict = {train_inputs: [word_index[text2]]})
# print(cosine_similarity(a, b))

# #   save lm model
# class LM(object):
#     def __init__(self, embed):
#         self.embed = embed

#     def get(self, w):
#         if (w in self.embed.keys()):
#             return self.embed[w]
#         return self.embed["<OOV>"]

# lm_embed = {}
# for i, w in enumerate(word_dictionary):
#     print(i, end = "\r")
#     lm_embed[w] = sess.run(embed, feed_dict = {train_inputs: [word_index[w]]})[0]
# lm_model = LM(lm_embed)
# print(cosine_similarity([lm_model.get(text1)], [lm_model.get(text2)]))
# with open("models/lm/embedding/model.pkl", "wb") as f:
#     pickle.dump(lm_model, f)

#   load lm model
# with open("models/lm/embedding/model.pkl", "rb") as f:
#     lm_model = pickle.load(f)
# print(cosine_similarity(lm_model.get(text1), lm_model.get(text2)))