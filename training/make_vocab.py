import numpy as np
import os

full_vocab = set()
vocab_files = os.listdir("data/RDR-wordlist")
for file in vocab_files:
	with open("data/RDR-wordlist/" + file, "r") as f:
		for row in f:
			full_vocab.add(row[:-1].lower())

with open("data/vietnamese-wordlist/Viet74K.txt", "r") as f:
	for row in f:
		full_vocab.add(row[:-1].lower())

full_vocab = list(full_vocab)
full_vocab.sort()

with open("data/full_vocab.txt", "w") as f:
	for w in full_vocab:
		f.write(w + "\n")