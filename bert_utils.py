import numpy as np
import tensorflow as tf

# Function to generate inputs. #
def generate_input(
    token_input, word_2_idx, vocab_size, 
    ker_sz, seq_length, out_length, CLS_token, 
    EOS_token, MSK_token, TRU_token, UNK_token):
    tmp_i_tok = [word_2_idx.get(
        x, UNK_token) for x in token_input]
    num_token = len(tmp_i_tok)
    
    # Truncate the sequence if it exceeds the maximum #
    # sequence length. Randomly select the review's   #
    # start and end index to be the positive example. #
    if num_token > seq_length:
        id_st = np.random.randint(
            0, num_token-seq_length)
        id_en = id_st + seq_length
        
        tmp_i_idx = [CLS_token]
        tmp_i_idx += tmp_i_tok[id_st:id_en]
        
        if id_en < num_token:
            # Add TRUNCATE token. #
            tmp_i_idx += [TRU_token]
        else:
            tmp_i_idx += [EOS_token]
        del id_st, id_en
    else:
        tmp_i_idx = [CLS_token]
        tmp_i_idx += tmp_i_tok
        tmp_i_idx += [EOS_token]
    
    # Generate the masked sequence. #
    n_input  = len(tmp_i_idx)
    mask_seq = [MSK_token] * n_input

    tmp_replace = [
        x for x in np.random.choice(
            vocab_size, size=n_input-2)]

    tmp_noise = [CLS_token]
    tmp_noise += tmp_replace
    tmp_noise += [tmp_i_idx[-1]]
    del tmp_replace
    
    # Sample within the average pooling kernel. #
    o_sample = np.random.randint(
        0, ker_sz, size=out_length+1)
    o_index  = [min(
        n_input-1, x+o_sample[int(x/ker_sz)]) \
            for x in range(0, n_input, ker_sz)]
    tmp_o_idx = [
        tmp_i_idx[x] for x in o_index[:out_length]]
    n_output  = len(tmp_o_idx)
    
    # Downsampling mask is all ones since every kernel will #
    # have one token masked out to train the embeddings.    #
    tmp_o_msk = [1.0 for _ in range(n_output)]
    
    # Set the input mask mechanism. #
    tmp_unif = np.random.uniform()
    if tmp_unif <= 0.8:
        # Replace with MASK token. #
        tmp_i_msk = [
            tmp_i_idx[x] if x not in o_index else \
                mask_seq[x] for x in range(n_input)]
    elif tmp_unif <= 0.9:
        # Replace with random word. #
        tmp_i_msk = [
            tmp_i_idx[x] if x not in o_index else \
                tmp_noise[x] for x in range(n_input)]
    else:
        # No replacement. #
        tmp_i_msk = tmp_i_idx
    return tmp_i_msk, tmp_o_idx, tmp_o_msk

# Sub-batch weight updates. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_mask, x_input, x_output, x_label, 
    optimizer, reg_cls=0.0, reg_emb=0.01, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_mask = x_mask[id_st:id_en, :]
        tmp_input  = x_input[id_st:id_en, :]
        tmp_label  = x_label[id_st:id_en]
        tmp_output = x_output[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model(
                tmp_input, training=True)
            
            class_logits = model_outputs[0]
            vocab_logits = model_outputs[1]
            bert_outputs = model_outputs[2]
            
            # Masked Language Model Loss. #
            msk_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=vocab_logits), tmp_mask)
            msk_losses = tf.reduce_sum(tf.reduce_sum(msk_xent, axis=1))
            
            # Embedding Loss for the CLS token since there is #
            # no Next Sentence Prediction (NSP) done.         #
            cls_embed = bert_outputs[:, 0, :]
            avg_embed = tf.reduce_mean(
                bert_outputs[:, 1:, :], axis=1)
            emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(cls_embed - avg_embed), axis=1))
            
            # Supervised Loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_label, logits=class_logits))
            
            # Full Loss Function. #
            pre_losses = msk_losses + reg_emb * emb_losses
            tmp_losses = tf.add(
                reg_cls * cls_losses, (1.0-reg_cls) * pre_losses)
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses

# Sub-batch weight updates for mix-up augmentation. #
def sub_batch_train_step_mixup(
    model, sub_batch_sz, 
    vocab_sz, n_class, x_mask1, x_mask2, 
    x_input1, x_input2, x_output1, x_output2, 
    x_label1, x_label2, optimizer, reg_cls=0.0, 
    reg_emb=0.01, learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input1.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_mask1 = x_mask1[id_st:id_en, :]
        tmp_mask2 = x_mask2[id_st:id_en, :]

        tmp_input1 = x_input1[id_st:id_en, :]
        tmp_label1 = x_label1[id_st:id_en]
        tmp_input2 = x_input2[id_st:id_en, :]
        tmp_label2 = x_label2[id_st:id_en]
        tmp_output1 = x_output1[id_st:id_en, :]
        tmp_output2 = x_output2[id_st:id_en, :]
        
        # Randomly set the mix-up proportion. #
        alpha = np.random.uniform(size=id_en-id_st)
        alpha = np.expand_dims(alpha, axis=1)
        
        tmp_mask = tf.add(
            alpha * tmp_mask1, (1.0-alpha) * tmp_mask2)
        tmp_mask = tf.cast(tmp_mask, tf.float32)
        tmp_label = tf.add(
            alpha * tf.one_hot(tmp_label1, n_class), 
            (1.0-alpha) * tf.one_hot(tmp_label2, n_class))
        
        alpha = np.expand_dims(alpha, axis=2)
        tmp_output = tf.add(
            alpha * tf.one_hot(tmp_output1, vocab_sz), 
            (1.0-alpha) * tf.one_hot(tmp_output2, vocab_sz))
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model.mixup_output(
                tmp_input1, tmp_input2, alpha, training=True)
            class_logits  = model_outputs[0]
            vocab_logits  = model_outputs[1]
            bert_enc_out  = model_outputs[2]
            
            # Masked Language Model Loss. #
            msk_xent = tf.multiply(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=vocab_logits), tmp_mask)
            num_mask = tf.cast(
                tf.reduce_sum(tmp_mask, axis=1), tf.float32)
            msk_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_xent, axis=1), num_mask))

            # CLS token embeddings. #
            cls_embed = bert_enc_out[:, 0, :]
            avg_embed = tf.reduce_mean(
                bert_enc_out[:, 1:, :], axis=1)
            emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(cls_embed - avg_embed), axis=1))

            # Supervised Loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tmp_label, logits=class_logits))
            
            # Full Loss Function. #
            pre_losses = msk_losses + reg_emb * emb_losses
            tmp_losses = tf.add(
                reg_cls * cls_losses, (1.0-reg_cls) * pre_losses)
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses