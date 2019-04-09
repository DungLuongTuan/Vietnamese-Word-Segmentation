import numpy as np 
from gensim.models import FastText
from gensim.models import KeyedVectors
import pdb

sentences = []

cnt = 0
with open("/home/common_gpu1/corpora/nlp/lm_corpus/thanhlc/line_corpus/full.txt", "r") as f:
    for row in f:
        cnt += 1
        print(cnt, end = "\r")
        sentences.append(row[:-1].split(" "))

print(len(sentences))
print("train model")
model = FastText(sentences, min_count = 5, size = 100, window = 10, iter = 20)

print("save model")
with open("models/word2vec/model.gs", "wb") as f:
    model.wv.save(f)

# model = FastText.load("models/word2vec/model.gen")
# similarities = model.wv.most_similar_cosmul(positive=['ăn'], negative=['interface'])
# print(similarities)

# model = KeyedVectors.load("models/word2vec/model.gs", mmap='r')
# similarities = model.most_similar_cosmul(positive=['Nơ'], negative=['interface'])
# print(similarities)
