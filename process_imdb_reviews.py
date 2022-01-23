import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter
from nltk.tokenize import wordpunct_tokenize

# Define the function to clean the data. #
def clean_data(data_df):
    n_lines = len(data_df)
    
    # Process the data. #
    w_counter = Counter()
    data_list = []
    for n_line in range(n_lines):
        data_line = data_df.iloc[n_line]
        data_text = data_line["text"]
        data_label = data_line["label"]
        data_score = data_line["rating"]
        
        # Clean the data a little. #
        data_text = data_text.replace("\n", " ")
        
        # Tokenize the words. #
        tmp_tokens = [
            x for x in wordpunct_tokenize(
                data_text.lower()) if x != ""]
        
        w_counter.update(tmp_tokens)
        data_list.append((
            data_label, data_score, tmp_tokens))
        
        if (n_line+1) % 5000 == 0:
            percent_complete = round(n_line / n_lines * 100, 2)
            print(str(n_line+1), "rows", 
                  "(" + str(percent_complete) + "%) complete.")
    return data_list, w_counter

# Define the mapping for the labels. #
label_2_id = {"pos": 0, "neg": 1}
id_2_label = {0: "pos", 1: "neg"}

# Load the data. #
print("Loading the data.")

tmp_path  = "/home/Data/imdb_movie_review/aclImdb/"
tmp_files = [x for x in os.listdir(tmp_path + "train/pos/")]

train_path  = tmp_path + "train/pos/"
train_tuple = []
for tmp_file in tmp_files:
    tmp_review = open(
        train_path + tmp_file).read()
    tmp_splits = tmp_file.split("_")
    tmp_fileid = tmp_splits[0]
    tmp_rating = tmp_splits[1]
    
    train_tuple.append(
        (tmp_fileid, 0, tmp_rating, tmp_review))

train_path = tmp_path + "train/neg/"
tmp_files  = [x for x in os.listdir(tmp_path + "train/neg/")]
for tmp_file in tmp_files:
    tmp_review = open(
        train_path + tmp_file).read()
    tmp_splits = tmp_file.split("_")
    tmp_fileid = tmp_splits[0]
    tmp_rating = tmp_splits[1]
    
    train_tuple.append(
        (tmp_fileid, 1, tmp_rating, tmp_review))

test_tuple = []
tmp_files  = [x for x in os.listdir(tmp_path + "test/pos/")]
test_path  = tmp_path + "test/pos/"
for tmp_file in tmp_files:
    tmp_review = open(
        test_path + tmp_file).read()
    tmp_splits = tmp_file.split("_")
    tmp_fileid = tmp_splits[0]
    tmp_rating = tmp_splits[1]
    
    test_tuple.append(
        (tmp_fileid, 0, tmp_rating, tmp_review))

test_path = tmp_path + "test/neg/"
tmp_files = [x for x in os.listdir(tmp_path + "test/neg/")]
for tmp_file in tmp_files:
    tmp_review = open(
        test_path + tmp_file).read()
    tmp_splits = tmp_file.split("_")
    tmp_fileid = tmp_splits[0]
    tmp_rating = tmp_splits[1]
    
    test_tuple.append(
        (tmp_fileid, 1, tmp_rating, tmp_review))

# Process the data. #
print("Processing training data.")

tmp_cols = ["fileid", "label", "rating", "text"]
train_df = pd.DataFrame(
    train_tuple, columns=tmp_cols)
test_df  = pd.DataFrame(
    test_tuple, columns=tmp_cols)
del train_tuple, test_tuple

tmp_output = clean_data(train_df)
train_data = tmp_output[0]
w_counter  = tmp_output[1]
del tmp_output

print("Processing test data.")
test_data  = clean_data(test_df)[0]

# Form the vocabulary. #
min_count  = 10
add_tokens = ["CLS", "UNK", "PAD", "EOS", "TRUNC"]
word_vocab = [
    word for word, count in \
    w_counter.items() if count >= min_count]
word_vocab = list(sorted(add_tokens + word_vocab))

idx_2_word = dict([(
    x, word_vocab[x]) for x in range(len(word_vocab))])
word_2_idx = dict([(
    word_vocab[x], x) for x in range(len(word_vocab))])
print("Vocab size:", len(word_vocab), "tokens.")

train_len = [len(x[2]) for x in train_data]
test_len  = [len(x[2]) for x in test_data]
print("95P Train Length:", np.quantile(train_len, 0.95))
print("95P Test Length: ", np.quantile(test_len, 0.95))

tmp_path = "/home/Data/imdb_movie_review/"
tmp_pkl_file = tmp_path + "train_movie_reviews_bert_data.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(train_data, tmp_file_save)
    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)

tmp_pkl_file = tmp_path + "test_movie_reviews_bert_data.pkl"
with open(tmp_pkl_file, "wb") as tmp_file_save:
    pkl.dump(test_data, tmp_file_save)
    pkl.dump(word_vocab, tmp_file_save)
    pkl.dump(idx_2_word, tmp_file_save)
    pkl.dump(word_2_idx, tmp_file_save)

