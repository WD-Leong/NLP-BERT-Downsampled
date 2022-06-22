import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_downsampled_keras as bert

from sklearn.metrics import classification_report
from bert_utils import generate_input, sub_batch_train_step

# Model Parameters. #
batch_size = 256
batch_test = 64
sub_batch  = 64
seq_length = 2000
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 1500
restore_flag  = False
save_step     = 50
warmup_steps  = 500
finetune_step = 800
final_tunings = 1300
display_step  = 1

ker_sz = 10
grad_clip  = 1.0
out_length = int((seq_length+2) / ker_sz)
prob_mask  = 0.15
prob_keep  = 0.90
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 50

model_ckpt_dir  = "TF_Models/movie_reviews_bert_downsampled_v1"
train_loss_file = "train_loss_movie_reviews_bert_downsampled_v1.csv"

# Load the data. #
tmp_path = "../../Data/movie_reviews/"
tmp_pkl_file = tmp_path + "movie_reviews_bert_data.pkl"
with open(tmp_pkl_file, "rb") as tmp_file_load:
    train_tuple = pkl.load(tmp_file_load)
    valid_tuple = pkl.load(tmp_file_load)
    
    word_vocab = pkl.load(tmp_file_load)
    idx_2_word = pkl.load(tmp_file_load)
    word_2_idx = pkl.load(tmp_file_load)

label_dict = dict([
    (0, "positive"), (1, "negative")])
vocab_size = len(word_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

num_test  = len(valid_tuple)
num_data  = len(train_tuple)
num_class = len(label_dict)

CLS_token = word_2_idx["CLS"]
EOS_token = word_2_idx["EOS"]
PAD_token = word_2_idx["PAD"]
UNK_token = word_2_idx["UNK"]
MSK_token = word_2_idx["MSK"]
TRU_token = word_2_idx["TRU"]
print("Total of", str(len(train_tuple)), "rows loaded.")

if num_test <= batch_test:
    n_val_batches = 1
elif num_test % batch_test == 0:
    n_val_batches = int(num_test / batch_test)
else:
    n_val_batches = int(num_test / batch_test) + 1

# Extract the test labels. #
test_labels = np.array([x[0] for x in valid_tuple])

# For the approximate validation accuracy. #
#rng = np.random.default_rng(12345)
idx = np.random.permutation(num_test)

n_approx_samples  = 200
acc_valid_batches = int(n_approx_samples / batch_test)
val_approx_labels = test_labels[idx[:n_approx_samples]]
print(np.mean(val_approx_labels))

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# Build the BERT Model. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERTClassifier(
    num_class, num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size, 
    seq_length+2, ker_sz, rate=1.0-prob_keep)
bert_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time()-start_time) / 60
print("BERT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optimizer=bert_optimizer)

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
else:
    print("Training a new model.")
    train_loss_list = []

# Print the model summary. #
tmp_init = np.zeros(
    [1, seq_length+2], dtype=np.int32)
tmp_pred = bert_model(tmp_init, training=True)

print(bert_model.summary())
del tmp_pred, tmp_init

# Train the BERT model. #
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_mask = np.zeros(
    [batch_size, out_length], dtype=np.float32)
tmp_out_seq = np.zeros(
    [batch_size, out_length], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    # Constant learning rate. #
    learning_rate = 1.0e-3

print("-" * 50)
print("Training the BERT Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print(str(num_test), "test data samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        # Constant learning rate. #
        learning_rate = 1.0e-3
    
    if n_iter >= finetune_step:
        reg_cls = 1.0

        if n_iter >= final_tunings:
            learning_rate = 1.0e-5
        else:
            learning_rate = 1.0e-4
    else:
        reg_cls = 0.0
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    print(batch_sample[:10])
    tmp_out_lab[:] = 0
    tmp_in_mask[:, :] = 1.0
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_tuple = train_tuple[tmp_index]
        tmp_label = tmp_tuple[0]
        tmp_score = tmp_tuple[1]
        tmp_i_tok = tmp_tuple[2].split(" ")

        tmp_seq_tuple = generate_input(
            tmp_i_tok, word_2_idx, 
            vocab_size, ker_sz, seq_length, 
            out_length, prob_mask, CLS_token, 
            EOS_token, MSK_token, TRU_token, UNK_token)
        
        n_input  = len(tmp_seq_tuple[0])
        n_output = len(tmp_seq_tuple[1])

        tmp_out_lab[n_index] = tmp_label
        tmp_in_seq[n_index, :n_input] = tmp_seq_tuple[0]
        tmp_out_seq[n_index, :n_output] = tmp_seq_tuple[1]
        tmp_in_mask[n_index, :n_output] = tmp_seq_tuple[2]
    
    # Set the training data. #
    tmp_in  = tmp_in_seq
    tmp_out = tmp_out_lab
    tmp_seq = tmp_out_seq
    tmp_msk = tmp_in_mask
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_msk, tmp_in, tmp_seq, tmp_out, 
        bert_optimizer, reg_cls=reg_cls, reg_emb=0.01, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)

    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        n_approx_batches = min(
            n_val_batches, acc_valid_batches+1)

        # Evaluate the model accuracy. #
        pred_labels = []
        for n_val_batch in range(n_approx_batches):
            id_st = n_val_batch * batch_test
            id_en = (n_val_batch+1) * batch_test
            
            if n_val_batch == (n_approx_batches-1):
                curr_batch = n_approx_samples - id_st
            else:
                curr_batch = batch_test
            
            tmp_test_tokens = np.zeros(
                [curr_batch, seq_length+2], dtype=np.int32)
            
            tmp_test_tokens[:, :] = PAD_token
            for tmp_n in range(curr_batch):
                test_idx  = idx[id_st + tmp_n]
                tmp_input = [
                    word_2_idx.get(x, UNK_token) for x in \
                        valid_tuple[test_idx][2].split(" ")]
                
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
        del tmp_pred_labels
        
        # Compute the accuracy. #
        n_correct = np.sum(np.where(
            pred_labels == val_approx_labels, 1, 0))
        accuracy  = float(n_correct) / n_approx_samples
        del pred_labels, n_correct
        
        end_tm = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("Test Accuracy:", str(round(accuracy*100, 2)) + "%.")
        
        train_loss_list.append((n_iter, avg_loss, accuracy))
        start_tm = time.time()
        print("-" * 50)
    
    # Model checkpoint. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_column = ["n_iter", "xent_loss", "test_acc"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_df_column)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

# Print the evaluation report for the full dataset. #
pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_test
    id_en = (n_val_batch+1) * batch_test
    
    if n_val_batch == (n_val_batches-1):
        curr_batch = num_test - id_st
    else:
        curr_batch = batch_test
    
    tmp_test_tokens = np.zeros(
        [curr_batch, seq_length+2], dtype=np.int32)
    
    tmp_test_tokens[:, :] = PAD_token
    for tmp_n in range(curr_batch):
        tmp_input = [
            word_2_idx.get(x, UNK_token) for x in \
                valid_tuple[id_st+tmp_n][2].split(" ")]
        
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
print("Generating evaluation report.")

pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
del tmp_pred_labels

pred_report = classification_report(
    test_labels, pred_labels, zero_division=0, 
    target_names=["positive", "negative"])
report_name = "movie_reviews_validation_report.txt"
with open(report_name, "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)
