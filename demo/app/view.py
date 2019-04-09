from app import app
from flask import render_template, request
from .apis import *
import tensorflow as tf
import numpy as np
import pickle

# paths
lm_model_path = "app/models/lm/embedding/model.pkl"
word_dictionary_path = "app/models/text/word_dictionary.txt"
char_dictionary_path = "app/models/text/char_dictionary.txt"
lexicon_path = "app/models/text/full_vocab.txt"
model_path = "app/models/model"

# parameters
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
#    model parameters
lr = 0.001
dropout = 0.8
batch_size = 256
num_epochs = 200

symbols = [".", ",", "!", "?"]
# load model
with open(lm_model_path, "rb") as f:
    lm = pickle.load(f)
### load word dictionary
word_dictionary = []
with open(word_dictionary_path, "r") as f:
    for row in f:
        word_dictionary.append(row[:-1])
print("word dictionary length: ", len(word_dictionary))

### load character dictionary
char_dictionary = []
with open(char_dictionary_path, "r") as f:
    for row in f:
        char_dictionary.append(row[:-1])
print("character dictionary length: ", len(char_dictionary))

### load vietnamese word list
vi_words = []
with open(lexicon_path, "r") as f:
    for row in f:
        vi_words.append(row[:-1])

feature_extractor = FeatureExtractor(lm, word_dictionary, char_dictionary, vi_words)
segmenter = DeepCNNBLSTMSegment(n_lstm_hidden, max_lstm_step, num_lstm_layers, word_dictionary, word_embedding_size, \
                                char_dictionary, character_embedding, max_word_length, filter_sizes, \
                                num_filters, initial_type, dict_order_size)
segmenter.load_model(model_path + "/model.ckpt")

def get_response(text):
    norm_text = ""
    for i, c in enumerate(text):
        if (i != 0) and ((text[i-1] in symbols) or ((c in symbols) and (text[i-1] != " "))):
            norm_text += " "
        norm_text += c
    result = {}
    feats = feature_extractor.transform(norm_text)
    segmented_text = segmenter.transform(norm_text, feats[0], feats[1], feats[2], feats[3])
    result["segmented_text"] = segmented_text
    print(norm_text)
    print(segmented_text)
    result = str(result).replace("'", '"')
    return result

#============================================= SERVER APPLICATION ========================================
@app.route('/')
def home():
    return render_template('main_page.html')

@app.route('/', methods = ['POST'])
def get_request():
    text = dict(request.form)['req'][0]
    res = get_response(text)
    print(res)
    return res