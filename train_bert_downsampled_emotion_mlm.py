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

# For the random replacement pre-training. #
special_tokens = [
    "CLS", "EOS", "PAD", "UNK", "MSK", "TRU"]

replace_vocab = [
    word_2_idx.get(x) for x in \
        word_vocab if x not in special_tokens]
vocab_replace = len(replace_vocab)

# Parameters. #
grad_clip  = 1.00
steps_max  = 10000
n_layers   = 3
n_heads    = 4
n_classes  = len(label_2_idx)
seq_length = 75

batch_size = 256
sub_batch  = 128
batch_test = 128

ker_sz = 3
p_keep = 0.90

hidden_size = 256
ffwd_size   = 4 * hidden_size
out_length  = int((seq_length+1) / ker_sz) + 1

warmup_steps  = 2500
finetune_step = 5000
final_tunings = 7500
cooling_step  = 500
save_step     = 100
display_step  = 50
restore_flag  = True

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

# Extract the labels. #
target_labels = [
    idx_2_label.get(x) for x in range(n_classes)]

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
    del train_loss_df
else:
    print("Training a new model.")
    train_loss_list = []

# Format the data before training. #
CLS_token = word_2_idx["CLS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
EOS_token = word_2_idx["EOS"]
MSK_token = word_2_idx["MSK"]
TRU_token = word_2_idx["TRU"]

# Approximate the validation performance. #
tuple_valid = valid_tuple
del valid_tuple

n_approx_val = 2500
valid_tuple  = [x for x in tuple_valid[:n_approx_val]]

num_valid = len(valid_tuple)
if num_valid <= batch_test:
    n_val_batches = 1
elif num_valid % batch_test == 0:
    n_val_batches = int(num_valid / batch_test)
else:
    n_val_batches = int(num_valid / batch_test) + 1

# Train the model. #
print("Training the BERT Model.")
print(str(len(train_tuple)), "training data.")
print(str(len(valid_tuple)), "validation data.")
print("-" * 50)

tot_loss = 0.0
num_data = len(train_tuple)

n_iter = ckpt.step.numpy().astype(np.int32)
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_mask = np.zeros(
    [batch_size, out_length], dtype=np.int32)
tmp_out_seq = np.zeros(
    [batch_size, out_length], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

start_tm = time.time()
while n_iter < steps_max:
    # Constant warmup rate. #
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_val
    
    if n_iter >= finetune_step:
        reg_cls = 1.0
        if n_iter < final_tunings:
            learning_rate = 1.0e-4
        else:
            learning_rate = 1.0e-5
    else:
        reg_cls = 0.0
    
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_lab[:] = 0
    tmp_in_mask[:, :] = 1.0
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = label_2_idx[train_tuple[tmp_index][0]]
        tmp_i_tok = train_tuple[tmp_index][1].split(" ")

        tmp_seq_tuple = generate_input(
            tmp_i_tok, word_2_idx, vocab_size, 
            ker_sz, seq_length, out_length, CLS_token, 
            EOS_token, MSK_token, TRU_token, UNK_token)
        
        n_input  = len(tmp_seq_tuple[0])
        n_output = len(tmp_seq_tuple[1])
        
        tmp_out_lab[n_index] = tmp_label
        tmp_in_seq[n_index, :n_input] = tmp_seq_tuple[0]
        tmp_in_mask[n_index, :n_output] = tmp_seq_tuple[2]
        tmp_out_seq[n_index, :n_output] = tmp_seq_tuple[1]
    
    # Set the training data. #
    tmp_in  = tmp_in_seq
    tmp_out = tmp_out_lab
    tmp_seq = tmp_out_seq
    tmp_msk = tmp_in_mask
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_msk, tmp_in, tmp_seq, tmp_out, 
        bert_optim, reg_cls=reg_cls, reg_emb=0.01, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # Get the validation accuracy. #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            if n_val_batch == (n_val_batches-1):
                id_en = num_valid
            else:
                id_en = (n_val_batch+1) * batch_test
            curr_batch = id_en - id_st

            tmp_test_tokens = np.zeros(
                [curr_batch, seq_length+2], dtype=np.int32)
            
            tmp_test_tokens[:, :] = PAD_token
            for tmp_n in range(curr_batch):
                curr_idx  = id_st + tmp_n
                tmp_input = [word_2_idx.get(
                    x, UNK_token) for x in \
                        valid_tuple[curr_idx][1].split(" ")]
                
                # Truncate if the length is longer. #
                n_input  = len(tmp_input)
                tmp_toks = [CLS_token]
                if n_input > seq_length:
                    tmp_toks += tmp_input[:seq_length]
                    tmp_toks += [TRU_token]
                else:
                    tmp_toks += tmp_input + [EOS_token]
                n_decode = len(tmp_toks)
                
                tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
                del tmp_toks, tmp_input, n_input, n_decode
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(
                tmp_test_tokens).numpy()
            pred_labels.append(tmp_pred_labels)
            del tmp_test_tokens
        
        # Concatenate the predicted labels. #
        pred_labels = np.concatenate(
            tuple(pred_labels), axis=0)
        
        # Compute the accuracy. #
        y_valid  = [
            label_2_idx[x] for x, y in valid_tuple]
        accuracy = np.sum(np.where(
            pred_labels == y_valid, 1, 0)) / num_valid
        
        # Generate the classification report. #
        eval_report = "bert_emotion_report.txt"
        eval_header = "BERT Word Token Classification Report"
        eval_header += " at iteration " + str(n_iter) + " \n"

        pred_report = classification_report(
            y_valid, pred_labels, 
            zero_division=0, target_names=target_labels)
        with open(eval_report, "w") as tmp_write:
            tmp_write.write(eval_header)
            tmp_write.write(pred_report)
        del pred_labels
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (time.time() - start_tm) / 60
        
        print("Iteration:", str(n_iter) + ".")
        print("Elapsed Time:", 
              str(round(elapsed_tm, 2)), "mins.")
        print("Learn Rate:", str(learning_rate) + ".")
        print("Average Train Loss:", str(avg_loss) + ".")
        print("Validation Accuracy:", 
              str(round(accuracy * 100, 2)) + "%.")
        
        start_tm = time.time()
        train_loss_list.append((n_iter, avg_loss, accuracy))
        
        if n_iter % cooling_step != 0:
            print("-" * 50)
    
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        df_col_names  = ["iter", "xent_loss", "val_acc"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=df_col_names)
        train_loss_df.to_csv(train_loss_file, index=False)
        del train_loss_df
    
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("-" * 50)

# Get the validation accuracy. #
num_test = len(tuple_valid)
if num_test <= batch_test:
    n_test_batches = 1
elif num_test % batch_test == 0:
    n_test_batches = int(num_test / batch_test)
else:
    n_test_batches = int(num_test / batch_test) + 1

pred_labels = []
for n_val_batch in range(n_test_batches):
    id_st = n_val_batch * batch_test
    if n_val_batch == (n_test_batches-1):
        id_en = num_test
    else:
        id_en = (n_val_batch+1) * batch_test
    curr_batch = id_en - id_st

    tmp_test_tokens = np.zeros(
        [curr_batch, seq_length+2], dtype=np.int32)
    
    tmp_test_tokens[:, :] = PAD_token
    for tmp_n in range(curr_batch):
        curr_idx  = id_st + tmp_n
        tmp_input = [
            word_2_idx.get(x, UNK_token) \
                for x in tuple_valid[curr_idx][1].split(" ")]
        
        # Truncate if the length is longer. #
        n_input  = len(tmp_input)
        tmp_toks = [CLS_token]
        if n_input > seq_length:
            tmp_toks += tmp_input[:seq_length]
            tmp_toks += [TRU_token]
        else:
            tmp_toks += tmp_input + [EOS_token]
        n_decode = len(tmp_toks)
        
        tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
        del tmp_toks, tmp_input, n_input, n_decode
    
    # Perform inference. #
    tmp_pred_labels = bert_model.infer(
        tmp_test_tokens).numpy()
    pred_labels.append(tmp_pred_labels)
    del tmp_test_tokens

# Concatenate the predicted labels. #
y_valid = [
    label_2_idx[x] for x, y in tuple_valid]
pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)

# Generate the classification report. #
eval_report = "bert_word_emotion_report.txt"
eval_header = "BERT Word Token Classification Report"
eval_header += " at iteration " + str(n_iter) + " \n"

pred_report = classification_report(
    y_valid, pred_labels, 
    zero_division=0, target_names=target_labels)
with open(eval_report, "w") as tmp_write:
    tmp_write.write(eval_header)
    tmp_write.write(pred_report)
del pred_labels
