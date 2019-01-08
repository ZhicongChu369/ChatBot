import tensorflow as tf
from preprocess import *
from tokenization import *
from model import *
import os
import numpy as np
import random
import codecs
import csv
import re
import unicodedata
from ast import literal_eval

##########PREPREOCESS AND PREPARE DATA##########

# Specify filepath of corpus
corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join(r'C:\Users\chuzh\Desktop\chatbot data', corpus_name)

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

load =load_corpus(corpus)
lines = load.loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
conversations =load.loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                         lines, MOVIE_CONVERSATIONS_FIELDS)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Write new csv file
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in load.extractSentencePairs(conversations):
        writer.writerow(pair)
        
# Load/Assemble voc and pairs
MAX_LEN = 10

trim_p = trim_pair(MAX_LEN)
save_dir = os.path.join("data", "save")
voc, pairs = trim_p.loadPrepareData(corpus, corpus_name, datafile, save_dir)

# Minimum word count threshold for trimming
MIN_COUNT = 3    
# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

# Get input and label txt as list of words
input_words, label_words, input_len, label_len = sep_input_label(pairs)

# Tokenize input and label
input_tokenized = tokenize( voc, input_words, window_size = MAX_LEN + 2)
label_tokenized = tokenize( voc, label_words, window_size = MAX_LEN + 2)

np.random.seed(0)
# shuffle data and split training and test set
idxes = np.random.permutation(len(input_tokenized))
thresh_num = int( len(input_tokenized)*0.8 )
idx_train = range(thresh_num )
idx_test = range( thresh_num+1, len(input_tokenized), 1)

train_input = [input_tokenized[i] for i in idx_train]
train_input_len = [input_len[i] for i in idx_train]
train_output = [label_tokenized[i] for i in idx_train]
train_output_len = [label_len[i] for i in idx_train]

test_input = [label_tokenized[i] for i in idx_test]
test_input_len = [input_len[i] for i in idx_test]
test_output = [label_tokenized[i] for i in idx_test]
test_output_len = [label_len[i] for i in idx_test]

# Creates batches. Remember that the Decoder labels need to be shifted over by 1.

dec_input_train = np.array(train_output)[:, 0: (MAX_LEN + 2 - 1)]
dec_input_test =  np.array(test_output)[:, 0: (MAX_LEN + 2 - 1)]
dec_label_train = np.array(train_output)[:, 1:]
dec_label_test =  np.array(test_output)[:, 1:]

def create_batches(data, batchSz):
    batches = []
    for i in range(len(data)//batchSz):
        batch = data[i*batchSz : (i+1)*batchSz] 
        batches.append(batch)
# add back the last mini batch        
    return(batches)
    
# Set Batch Size
BATCH_SIZE = 128

train_enc_baches = np.array(create_batches(train_input, BATCH_SIZE))
train_input_len_baches = np.array(create_batches(train_input_len, BATCH_SIZE))
train_dec_input_baches = np.array(create_batches(dec_input_train, BATCH_SIZE))
train_dec_label_baches = np.array(create_batches(dec_label_train, BATCH_SIZE))
train_output_len_baches = np.array(create_batches(train_output_len, BATCH_SIZE))

# Test batch size set to 1000
test_enc_baches = create_batches(test_input, 1000)
test_input_len_baches = create_batches(test_input_len, 1000)
test_dec_input_baches = create_batches(dec_input_test, 1000)
test_dec_label_baches = create_batches(dec_label_test, 1000)
test_output_len_baches = create_batches(test_output_len, 1000)

tf.reset_default_graph()

# Initialize model
model = Model( MAX_LEN = 10, vocab_size = len(voc.word2index))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

steps = len(train_input)//BATCH_SIZE
#steps = 20
EPOC = 20
for i in range(EPOC):
    print('Training EPOC %d....'%(i))
    for step in range(steps):
        _,loss = sess.run([model.train, model.loss], 
                                 feed_dict = {model.encoder_input : train_enc_baches[step],
                                              model.encoder_input_length: train_input_len_baches[step],
                                              model.decoder_input :train_dec_input_baches[step],
                                              model.decoder_input_length : train_output_len_baches[step],
                                              model.decoder_labels : train_dec_label_baches[step],
                                              model.keep_prob: 0.8}  )
#    print(loss)
    
steps = len(test_input)//1000
#steps = 20
total_loss = 0
total_num_words = 0
accurate_words = 0
for step in range(steps):
    loss, acc_words = sess.run([model.loss, model.accWords],
                             feed_dict = {model.encoder_input : test_enc_baches[step],
                                          model.encoder_input_length: test_input_len_baches[step],
                                          model.decoder_input :test_dec_input_baches[step],
                                          model.decoder_input_length : test_output_len_baches[step],
                                          model.decoder_labels : test_dec_label_baches[step],
                                          model.keep_prob: 1}  )
    #print(loss)
    num_words = np.sum(test_output_len[step*1000 : (step+1)*1000]) - 1000
    batch_loss = loss*num_words
    total_loss += batch_loss
    total_num_words += num_words
    accurate_words += acc_words
    #print(log.shape)
    #print(acc.shape)
    #print(acc)
    
    perpl = np.exp(total_loss/total_num_words)
accuracy = accurate_words / total_num_words
print(perpl)
print(accuracy)
