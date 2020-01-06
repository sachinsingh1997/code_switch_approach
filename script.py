#author: Sachin_singh
# coding: utf-8

import numpy as np
import pandas as pd
import csv
import re
import sys
import os

train_file = str(sys.argv[1])
test_file = str(sys.argv[2])

os.system("sed -i '/\"/d' ./"+str(train_file))
os.system("sed -i '/\"/d' ./"+str(test_file))

# Had to remove '"' entries from the 'train_data.tsv' file as they interfere in the reading/parsing of read_csv
# process.




# Reading the training file
names = ['tweetID', 'userID', 'st', 'ed', 'token', 'tag']
df = pd.read_csv(train_file, delimiter='\t', names = names, header=None)
print(df.tail())
print()
print("Train size:", len(df))

# reading the testing file
names = ['tweetID', 'userID', 'st', 'ed', 'token', 'tag']
df1 = pd.read_csv(test_file, delimiter='\t', names = names, header=None)
print(df1.tail())
print()
print("Test size:", len(df1))


# Concatinating both for getting total words and chars embeddings.
dt = [df, df1]
D = pd.concat(dt)
print("Total Data:")
print(D.tail())

# Total words in the corpus (train+test)
words = list(set(D["token"].values))
n_words = len(words)
print(n_words)


# number of tags
tags = list(set(D["tag"].values))
n_tags = len(tags)
print("Number of tags", n_tags) #es, en, other


# Creatinng Training data
t_id = df['tweetID'][0]
u_id = df['userID'][0]
tup = ()
sents_train = []
tmp_sent = []

for i in range(len(df)):
    if(t_id == df['tweetID'][i] and u_id == df['userID'][i]):
        tup = (str(df['token'][i]), str(df['tag'][i]))
        tmp_sent.append(tup)
    else:
        t_id = df['tweetID'][i]
        u_id = df['userID'][i]
        sents_train.append(tmp_sent)
        tmp_sent = []
        tup = (str(df['token'][i]), str(df['tag'][i]))
        tmp_sent.append(tup)


# Creating Testing data
t_id = df1['tweetID'][0]
u_id = df1['userID'][0]
tup = ()
sents_test = []
tmp_sent = []

for i in range(len(df)):
    if(t_id == df['tweetID'][i] and u_id == df['userID'][i]):
        tup = (str(df['token'][i]), str(df['tag'][i]))
        tmp_sent.append(tup)
    else:
        t_id = df['tweetID'][i]
        u_id = df['userID'][i]
        sents_test.append(tmp_sent)
        tmp_sent = []
        tup = (str(df['token'][i]), str(df['tag'][i]))
        tmp_sent.append(tup)

print("Train sents len: ", len(sents_train))
print("Test Sents len: ", len(sents_test))


# Fixing paramenters for future refereces
max_len = 75
max_len_char = 10


# Creaing dictionaries for words, tags as well as chars
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}


# Testing for the indexes
print(word2idx["where"])
print("En: ", tag2idx["en"])
print("Es: ", tag2idx["es"])
print("Other: ", tag2idx["other"])


# Creating word embeddings for train and test
X_word_tr = [[word2idx[w[0]] for w in s] for s in sents_train]
X_word_te = [[word2idx[w[0]] for w in s] for s in sents_test]

# Padding sequences to make all of same size
from keras.preprocessing.sequence import pad_sequences

# Padded word embeddings for traina and test words
X_word_tr = pad_sequences(maxlen=max_len, sequences=X_word_tr, value=word2idx["PAD"], padding='post', truncating='post')
X_word_te = pad_sequences(maxlen=max_len, sequences=X_word_te, value=word2idx["PAD"], padding='post', truncating='post')

# Total chars in the corpus
chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print("Total characters: ", n_chars)


char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0


# train
X_char_tr = []
for sentence in sents_train:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char_tr.append(np.array(sent_seq))
    
# test
X_char_te = []
for sentence in sents_test:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char_te.append(np.array(sent_seq))


# Tags set for training and testing
y_tr = [[tag2idx[w[1]] for w in s] for s in sents_train]
y_te = [[tag2idx[w[1]] for w in s] for s in sents_test]



# Padding the tags sequence
y_tr = pad_sequences(maxlen=max_len, sequences=y_tr, value=tag2idx["PAD"], padding='post', truncating='post')
y_te = pad_sequences(maxlen=max_len, sequences=y_te, value=tag2idx["PAD"], padding='post', truncating='post')


# Importing modules for our DNN model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D


# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=max_len, mask_zero=True)(word_in)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)
out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

model = Model([word_in, char_in], out)


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_crossentropy"])

print("Model Summary: ")
print(model.summary())

from keras.models import load_model

# fixing parameters for our BiLSTM model

print("Starting the training of our model......")
epochs = 10
batch_size = 256
history = model.fit([X_word_tr,
                     np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                    np.array(y_tr).reshape(len(y_tr), max_len, 1),
                    batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)

model.save('BiLSTMmodel.hdf5')

hist = pd.DataFrame(history.history)

# Displaying the variations of our Loss as well as validation loss while training

import matplotlib.pyplot as plt
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])
plt.show()

# Prediction using our trained Model
print("Initiating our prediction of test data...............")
y_pred = model.predict([X_word_te,
                        np.array(X_char_te).reshape((len(X_char_te),
                                                     max_len, max_len_char))])


# Printing the true and predicted tags for the test tokesns 
# This is also important as we'll derive our classifiction report as well as confusion matrix from
# this result only

print(30 * "-")
print("Comparision of true vs predicted tags")
print(30 * "-")

print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")

Y1 = []
Y2 = []

for i in range(len(y_pred)):
    p = np.argmax(y_pred[i], axis=-1)
    for w, t, pred in zip(X_word_te[i], y_te[i], p):
        if w != 0:
            print("{:15}: {:5} {}".format(idx2word[w], idx2tag[t], idx2tag[pred]))
            Y1.append(idx2tag[t])
            Y2.append(idx2tag[pred])


print("Counts of different tags in Test set")
print("En: ", Y1.count('en'))
print("Es: ", Y1.count('es'))
print("Other: ", Y1.count('other'))


k = len(Y1)
# These are computed as to be used to check our models performance
P1 = np.zeros((k,3), dtype=np.float32)
P2 = np.zeros((k,3), dtype=np.float32)

for i in range(k):
    if Y1[i]=='other':
        P1[i][0]=1.0
    elif Y1[i]=='en':
        P1[i][1]=1.0
    elif Y1[i]=='es':
        P1[i][2]=1.0
        
for i in range(k):
    if Y2[i]=='other':
        P2[i][0]=1.0
    elif Y2[i]=='en':
        P2[i][1]=1.0
    elif Y2[i]=='es':
        P2[i][2]=1.0

# print(P1, P2)


from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix for our prediction model
P1 = np.argmax(P1, axis=1)
for ix in range(3):
    print(ix, confusion_matrix(np.argmax(P2, axis=1), P1)[ix].sum())
cm = confusion_matrix(np.argmax(P2, axis=1), P1)
print(cm)

import seaborn as sn
import matplotlib.pyplot as plt
 
# An image of our confusion matrix
df_cm = pd.DataFrame(cm, range(3), range(3))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=False)
sn.set_context("poster")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# for computng the classification report as well as the accuracy of our model.
target_names = ['other', 'en', 'es']
print("Classification report for our model running on the test file")
print(classification_report(np.argmax(P2, axis=1), P1, target_names=target_names))
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(np.argmax(P2, axis=1), P1, normalize=True))

