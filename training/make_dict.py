dictionary = set()
with open("data/dictionary_freq.txt", "r") as f:
	for row in f:
		split_row = row[:-1].split("\t")
		if (int(split_row[1]) > 3):
			dictionary.add(split_row[0])
			dictionary.add(split_row[0].lower())
dictionary = list(dictionary)
dictionary.sort()
print("dictionary length: ", len(dictionary))
dictionary.append("<OOV>")
with open("data/lower_word_dictionary.txt", "w") as f:
	for word in dictionary:
		f.write(word + "\n")
