import pickle
import numpy as np
import pdb

# class LM(object):
#     def __init__(self, embed):
#         self.embed = embed

#     def get(self, w):
#         if (w in self.embed.keys()):
#             return self.embed[w]
#         return self.embed["<OOV>"]


class FeatureExtractor(object):
    def __init__(self, lm, word_dictionary, char_dictionary, vi_words):
        self.lm = lm
        ### load word dictionary
        self.word_dictionary = word_dictionary
        ### load character dictionary
        self.char_dictionary = char_dictionary
        ### load vietnamese word list
        self.vi_words = vi_words
        ### model params
        self.max_word_length = 10
        self.max_lstm_step = 250

    def transform(self, sentence):
        word_data = []
        char_data = []
        dict_order_feat = []
        words = sentence.split(" ")
        seqlen = len(words)
        for word in words:
            word_data.append(self.lm.get(word.lower()))
            charvec = []
            for char in word:
                if (char in self.char_dictionary):
                    charvec.append(self.char_dictionary.index(char))
                else:
                    charvec.append(self.char_dictionary.index("<OOV>"))
            while (len(charvec) < self.max_word_length):
                charvec.append(self.char_dictionary.index("<PAD>"))
            char_data.append(charvec[:min(self.max_word_length, len(charvec))])
        while len(word_data) < self.max_lstm_step:
            word_data.append(np.zeros(100))
            charvec = list(np.ones(self.max_word_length)*self.char_dictionary.index("<PAD>"))
            char_data.append(charvec)

        words = ["none", "none"] + words + ["none", "none"]
        for i in range(2, len(words) - 2):
            feat = [0, 0, 0, 0, 0]
            if (' '.join(words[i-2:i+1]) in self.vi_words) or (' '.join(words[i-2:i+1]).lower() in self.vi_words):
                feat[0] = 1
            if (' '.join(words[i-1:i+1]) in self.vi_words) or (' '.join(words[i-1:i+1]).lower() in self.vi_words):
                feat[1] = 1
            if (' '.join(words[i-1:i+2]) in self.vi_words) or (' '.join(words[i-1:i+2]).lower() in self.vi_words):
                feat[2] = 1
            if (' '.join(words[i:i+2]) in self.vi_words) or (' '.join(words[i:i+2]).lower() in self.vi_words):
                feat[3] = 1
            if (' '.join(words[i:i+3]) in self.vi_words) or (' '.join(words[i:i+3]).lower() in self.vi_words):
                feat[4] = 1
            dict_order_feat.append(feat)

        while (len(dict_order_feat) < self.max_lstm_step):
            dict_order_feat.append([0, 0, 0, 0, 0])

        return word_data[:min(self.max_lstm_step, len(word_data))], char_data[:min(self.max_lstm_step, len(char_data))], dict_order_feat[:min(self.max_lstm_step, len(dict_order_feat))], seqlen

# def main():
#     lm_model_path = "/home/tittit/python/graduate_project/submit/demo/app/models/lm/embedding/model.pkl"
#     word_dictionary_path = "/home/tittit/python/graduate_project/submit/demo/app/models/text/word_dictionary.txt"
#     char_dictionary_path = "/home/tittit/python/graduate_project/submit/demo/app/models/text/char_dictionary.txt"
#     lexicon_path = "/home/tittit/python/graduate_project/submit/demo/app/models/text/full_vocab.txt"

#     with open(lm_model_path, "rb") as f:
#         lm = pickle.load(f)
#     ### load word dictionary
#     word_dictionary = []
#     with open(word_dictionary_path, "r") as f:
#         for row in f:
#             word_dictionary.append(row[:-1])
#     print("word dictionary length: ", len(word_dictionary))

#     ### load character dictionary
#     char_dictionary = []
#     with open(char_dictionary_path, "r") as f:
#         for row in f:
#             char_dictionary.append(row[:-1])
#     print("character dictionary length: ", len(char_dictionary))

#     ### load vietnamese word list
#     vi_words = []
#     with open(lexicon_path, "r") as f:
#         for row in f:
#             vi_words.append(row[:-1])

#     feature_extractor = FeatureExtractor(lm, word_dictionary, char_dictionary, vi_words)
#     text = "Từ những năm 1990 , Nguyễn Thái Tài đã được nhiều người biết tới do được xem là nam giới đầu tiên hành nghề trang điểm chuyên nghiệp cho phái đẹp , góp phần đưa nhiều người mẫu , hoa khôi , hoa hậu lên bục vinh quang ."
#     feats = feature_extractor.transform(text)
#     pdb.set_trace()


# if __name__ == '__main__':
#     main()