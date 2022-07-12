# NLP-BERT-Downsampled
This repository contains a simple implementation to extend the [BERT model](https://arxiv.org/abs/1810.04805) to process longer sequences by applying average pooling of the sequence before applying the self-attention mechanism. It has two modes - (i) the model trains without any pre-training by using only the `CLS` output embeddings to train the classifier, and (ii) the model applies Masked Language Model (MLM) pre-training to generate the word embeddings prior to fine-tuning. Please note that the codes are still being refined.

## Training the Classifier
To train the classifier, run
```
python train_downsampled_emotion_mlm.py
```
and
```
python infer_downsampled_emotion_mlm.py
```
to perform inference. For the movie review dataset, the model achieves 
```
BERT Word Token Classification Report at iteration 1500 
              precision    recall  f1-score   support

    positive       0.90      0.88      0.89     12500
    negative       0.89      0.90      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000
```
and
```
BERT Word Token Classification Report at iteration 10000 
              precision    recall  f1-score   support

       anger       0.91      0.92      0.92     14320
        fear       0.88      0.86      0.87     11949
         joy       0.94      0.93      0.94     35285
        love       0.82      0.83      0.82      8724
     sadness       0.95      0.96      0.95     30161
    surprise       0.76      0.82      0.79      3764

    accuracy                           0.92    104203
   macro avg       0.88      0.89      0.88    104203
weighted avg       0.92      0.92      0.92    104203
```
on the emotion dataset.
