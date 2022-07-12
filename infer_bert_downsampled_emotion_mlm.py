import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_downsampled_keras as bert
from bert_utils import (
    sub_batch_train_step, generate_input)
from sklearn.metrics import classification_report

# Load the data. #
print("Loading the data.")

tmp_path = "/home/Data/emotion_dataset/"
tmp_pkl_file = tmp_path + "emotion_word_bert.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    train_tuple = pkl.load(tmp_file_load)
    valid_tuple = pkl.load(tmp_file_load)

    word_vocab = pkl.load(tmp_file_load)
    idx_2_word = pkl.load(tmp_file_load)
    word_2_idx = pkl.load(tmp_file_load)
    
    label_2_idx = pkl.load(tmp_file_load)
    idx_2_label = pkl.load(tmp_file_load)

vocab_size = len(word_vocab)
print("Total of", vocab_size, "tokens.")

# Parameters. #
n_layers   = 3
n_heads    = 4
n_classes  = len(label_2_idx)
seq_length = 75

ker_sz = 3
p_keep = 0.90

hidden_size = 256
ffwd_size   = 4 * hidden_size
out_length  = int((seq_length+1) / ker_sz) + 1

model_ckpt_dir  = "TF_Models/bert_downsampled_emotion_model"
train_loss_file = "train_loss_emotion_bert_downsampled.csv"

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Define the classifier model. #
bert_model = bert.BERTClassifier(
    n_classes, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length+2, ker_sz, rate=1.0-p_keep)
bert_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Model restored from {}".format(
        manager.latest_checkpoint))
else:
    print("Error: No latest checkpoint found.")
n_iter = ckpt.step.numpy().astype(np.int32)

# Extract the special tokens. #
CLS_token = word_2_idx["CLS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
EOS_token = word_2_idx["EOS"]
MSK_token = word_2_idx["MSK"]
TRU_token = word_2_idx["TRU"]

# Model Inference. #
print("Inference of the BERT Model.")
print("(" + str(n_iter), "iterations.)")
print("-" * 50)

while True:
    tmp_phrase = input("Enter text: ")
    tmp_phrase = tmp_phrase.lower().strip()
    
    if tmp_phrase == "":
        break
    else:
        tmp_test_tokens = np.zeros(
            [1, seq_length+2], dtype=np.int32)
        tmp_test_tokens[:, :] = PAD_token
        
        tmp_input = [
            word_2_idx.get(x, UNK_token) \
                for x in tmp_phrase.split(" ")]
        
        # Truncate if the length is longer. #
        n_input  = len(tmp_input)
        tmp_toks = [CLS_token]
        if n_input > seq_length:
            tmp_toks += tmp_input[:seq_length]
            tmp_toks += [TRU_token]
        else:
            tmp_toks += tmp_input + [EOS_token]
        n_decode = len(tmp_toks)
        
        tmp_test_tokens[0, :n_decode] = tmp_toks
        del tmp_toks, tmp_input, n_input, n_decode
            
        # Perform inference. #
        tmp_pred_label = bert_model.infer(
            tmp_test_tokens).numpy()[0]
        del tmp_test_tokens
        
        print("Input Text:", tmp_phrase)
        print("Pred Label:", idx_2_label[tmp_pred_label])
        print("-" * 50)

