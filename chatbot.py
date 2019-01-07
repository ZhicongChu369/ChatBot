from preprocess import *
from tokenization import *
import model
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
train_output = [label_tokenized[i] for i in idx_train]
test_input = [label_tokenized[i] for i in idx_test]
test_output = [label_tokenized[i] for i in idx_test]

# Creates batches. Remember that the Decoder labels need to be shifted over by 1.

dec_input_train = train_output[:, 0: (MAX_LEN + 2 - 1)]
dec_input = train_output[:, 0: (MAX_LEN + 2 - 1)]

def create_batches(data, batchSz, windowSz):
    batches = []
    for i in range(len(data)//batchSz):
        batch = data[i*batchSz : (i+1)*batchSz] 
        batches.append(batch)
# add back the last mini batch        
    return(batches)
    
# Set Batch Size
BATCH_SIZE = 128

train_input_baches = create_batches(train_input, BATCH_SIZE, MAX_LEN + 2)
train_label_baches = create_batches(train_label, BATCH_SIZE, MAX_LEN + 2)

# Test batch size set to 1000
test_input_baches = create_batches(test_input, 1000, MAX_LEN + 2)
test_label_baches = create_batches(test_label, 1000, MAX_LEN + 2)




