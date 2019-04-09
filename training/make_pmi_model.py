import os
import numpy as np 
import pickle
import pdb
import math

data_path = "/home/common/corpora/nlp/lm_corpus/thanhlc/line_corpus/full.txt"

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

#   parameters
min_freq = 15

#   load dictionary
# print("load dictionary")
# word_dictionary = []
# cnt = 0
# for word in dictionary:
#     cnt += 1
#     print(cnt, end = "\r")
#     if dictionary_freq[word] > min_freq:
#         word_dictionary.append(word)
# word_dictionary.sort()
# word_index = {}
# for i, word in enumerate(word_dictionary):
#     word_index[word] = i
# print(len(word_dictionary))
# print(len(word_dictionary))

# #   make dependency matrix
# if not os.path.exists('models/pmi/dependency_matrix.npy'):
#     dependency_matrix = np.zeros((len(word_dictionary), len(word_dictionary)), np.int32)
#     cnt = 0
#     with open(data_path, 'r') as f:
#         for row in f:
#             cnt += 1
#             print(cnt, end = "\r")
#             words = row[:-1].split(" ")
#             for i in range(len(words) - 1):
#                 if (words[i] in word_index.keys()) and (words[i+1] in word_index.keys()):
#                     dependency_matrix[word_index[words[i]], word_index[words[i+1]]] += 1
#     np.save('models/pmi/dependency_matrix.npy', dependency_matrix)
# else:
#     dependency_matrix = np.load('models/pmi/dependency_matrix.npy')

# pdb.set_trace()

word_dictionary = []

with open('/home/dunglt_gpu0/graduate_project/data/word_dictionary.txt', 'r') as f:
    for row in f:
        word_dictionary.append(row[:-1])
word_index = {}
for i, word in enumerate(word_dictionary):
    word_index[word] = i

sum_words = 0
with open(data_path, 'r') as f:
    for row in f:
        words = row[:-1].split(" ")
        sum_words += len(words)

print('make dependency dictionary')
if not os.path.exists('models/pmi/dependency_dict.pkl'):
    dependency_dict = {}
    cnt = 0
    with open(data_path, 'r') as f:
        for row in f:
            cnt += 1
            print(cnt , end = "\r")
            words = row[:-1].split(" ")
            for i in range(len(words) - 1):
                if (words[i] in word_index.keys()) and (words[i+1] in word_index.keys()):
                    if ' '.join([words[i], words[i+1]]) in dependency_dict.keys():
                        dependency_dict[' '.join([words[i], words[i+1]])] += int(1)
                    else:
                        dependency_dict[' '.join([words[i], words[i+1]])] = int(1)
            for i in range(len(words) - 2):
                if (words[i] in word_index.keys()) and (words[i+1] in word_index.keys() and (words[i+2] in word_index.keys())):
                    if ' '.join([words[i], words[i+1], words[i+2]]) in dependency_dict.keys():
                        dependency_dict[' '.join([words[i], words[i+1], words[i+2]])] += int(1)
                    else:
                        dependency_dict[' '.join([words[i], words[i+1], words[i+2]])] = int(1)

    with open('models/pmi/dependency_dict.pkl', 'wb') as f:
        pickle.dump(dependency_dict, f)
else:
    with open('models/pmi/dependency_dict.pkl', 'rb') as f:
        dependency_dict = pickle.load(f)

print('make jump dependency dictionary')
if not os.path.exists('models/pmi/jump_dependency_dict.pkl'):
    jump_dependency_dict = {}
    cnt = 0
    with open(data_path, 'r') as f:
        for row in f:
            cnt += 1
            print(cnt , end = "\r")
            words = row[:-1].split(" ")
            for i in range(len(words) - 2):
                if (words[i] in word_index.keys()) and (words[i+2] in word_index.keys()):
                    if ' '.join([words[i], words[i+2]]) in jump_dependency_dict.keys():
                        jump_dependency_dict[' '.join([words[i], words[i+2]])] += int(1)
                    else:
                        jump_dependency_dict[' '.join([words[i], words[i+2]])] = int(1)

    with open('models/pmi/jump_dependency_dict.pkl', 'wb') as f:
        pickle.dump(jump_dependency_dict, f)
else:
    with open('models/pmi/jump_dependency_dict.pkl', 'rb') as f:
        jump_dependency_dict = pickle.load(f)

max_lstm_step = 250
INF = 1000000000

def ext_tanh(x):
    return (1 - np.exp(-0.5*x))/(1 + np.exp(-0.5*x))

def gen_features(inline_path):
    cnt = 0
    dict_order_feats = []
    with open(inline_path, "r") as f:
        for row in f:
            cnt += 1
            # print(cnt, end = "\r")
            # print(row[:-1])
            dict_order_feat = []
            words = row[:-1].split(" ")
            words = ["none", "none"] + words + ["none", "none"]
            for i in range(2, len(words) - 2):
                feat = [0, 0, 0, 0, 0]
                if (' '.join(words[i-2:i+1]).lower() in dependency_dict.keys()):
                    x = dictionary_freq[words[i-2].lower()]/sum_words
                    y = dictionary_freq[words[i-1].lower()]/sum_words
                    z = dictionary_freq[words[i].lower()]/sum_words
                    xy = dependency_dict[' '.join(words[i-2:i]).lower()]/sum_words
                    yz = dependency_dict[' '.join(words[i-1:i+1]).lower()]/sum_words
                    xz = jump_dependency_dict[' '.join([words[i-2], words[i]]).lower()]/sum_words
                    xyz = dependency_dict[' '.join(words[i-2:i+1]).lower()]/sum_words
                    feat[0] = max(math.log((xy*yz*xz)/(xyz*x*y*z)), 0)
                    if feat[0] > 4:
                        feat[0] = 1
                if (' '.join(words[i-1:i+1]).lower() in dependency_dict.keys()):
                    x = dictionary_freq[words[i-1].lower()]/sum_words
                    y = dictionary_freq[words[i].lower()]/sum_words
                    xy = dependency_dict[' '.join(words[i-1:i+1]).lower()]/sum_words
                    feat[1] = max(math.log(xy/(x*y)), 0)
                    if feat[1] > 4:
                        feat[1] = 1
                if (' '.join(words[i-1:i+2]).lower() in dependency_dict.keys()):
                    x = dictionary_freq[words[i-1].lower()]/sum_words
                    y = dictionary_freq[words[i].lower()]/sum_words
                    z = dictionary_freq[words[i+1].lower()]/sum_words
                    xy = dependency_dict[' '.join(words[i-1:i+1]).lower()]/sum_words
                    yz = dependency_dict[' '.join(words[i:i+2]).lower()]/sum_words
                    xz = jump_dependency_dict[' '.join([words[i-1], words[i+1]]).lower()]/sum_words
                    xyz = dependency_dict[' '.join(words[i-1:i+2]).lower()]/sum_words
                    feat[2] = max(math.log((xy*yz*xz)/(xyz*x*y*z)), 0)
                    if feat[2] > 4:
                        feat[2] = 1
                if (' '.join(words[i:i+2]).lower() in dependency_dict.keys()):
                    x = dictionary_freq[words[i].lower()]/sum_words
                    y = dictionary_freq[words[i+1].lower()]/sum_words
                    xy = dependency_dict[' '.join(words[i:i+2]).lower()]/sum_words
                    feat[3] = max(math.log(xy/(x*y)), 0)
                    if feat[3] > 4:
                        feat[3] = 1
                if (' '.join(words[i:i+3]).lower() in dependency_dict.keys()):
                    x = dictionary_freq[words[i].lower()]/sum_words
                    y = dictionary_freq[words[i+1].lower()]/sum_words
                    z = dictionary_freq[words[i+2].lower()]/sum_words
                    xy = dependency_dict[' '.join(words[i:i+2]).lower()]/sum_words
                    yz = dependency_dict[' '.join(words[i+1:i+3]).lower()]/sum_words
                    xz = jump_dependency_dict[' '.join([words[i], words[i+2]]).lower()]/sum_words
                    xyz = dependency_dict[' '.join(words[i:i+3]).lower()]/sum_words
                    feat[4] = max(math.log((xy*yz*xz)/(xyz*x*y*z)), 0)
                    if feat[4] > 4:
                        feat[4] = 1
                dict_order_feat.append(list(np.array(feat)))
                # print(words[i-2:i+3], "\t", feat)
                # dict_order_feat.append(list(ext_tanh(np.array(feat))))
            while (len(dict_order_feat) < max_lstm_step):
                dict_order_feat.append([0, 0, 0, 0, 0])
            dict_order_feats.append(dict_order_feat[:max_lstm_step])
    return dict_order_feats

print('load train_dict_order_feats')
train_dict_order_feats = gen_features('data/VTB-train-dev-test/train-inline')
print(np.shape(train_dict_order_feats))
with open('data/norm_pmi/train_dict_order_feats.pkl', 'wb') as f:
    pickle.dump(train_dict_order_feats, f)

print('load dev_dict_order_feats')
dev_dict_order_feats = gen_features('data/VTB-train-dev-test/dev-inline')
print(np.shape(dev_dict_order_feats))
with open('data/norm_pmi/dev_dict_order_feats.pkl', 'wb') as f:
    pickle.dump(dev_dict_order_feats, f)

print('load test_dict_order_feats')
test_dict_order_feats = gen_features('data/VTB-train-dev-test/test-inline')
print(np.shape(test_dict_order_feats))
with open('data/norm_pmi/test_dict_order_feats.pkl', 'wb') as f:
    pickle.dump(test_dict_order_feats, f)
