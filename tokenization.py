def sep_input_label(paired_txt):
    '''Separate input and label sentences from paired txt'''
    input_words = [ ]
    label_words = [ ]
    input_len = [ ]
    label_len = [ ]
    for i in range( len(paired_txt) ):
        
        input_txt, label_txt = paired_txt[i][0], paired_txt[i][1]
        input_words_cur = input_txt.split()
        label_words_cur = label_txt.split()
        input_words.append(input_words_cur)
        label_words.append(label_words_cur)
        
        # original length +2 to take SOS and EOS into consideration
        input_len.append( len(input_words_cur) + 2)
        label_len.append( len(input_words_cur) + 2)
        
    return input_words, label_words, input_len, label_len
        
def indexesFromSentence(voc, sentence_words):
    '''Transform words to index and add Start and END tokens'''
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    return [SOS_token] + [voc.word2index[word] for word in sentence_words] + [EOS_token]         

def zeroPadding( sentence_token, window_size ):
    '''Pad tokenized sentence to the length of window size (excluding start and end sentence token)'''
    PAD_token = 0  # Used for padding short sentences
    sentence_token_padded = sentence_token
    if len(sentence_token) < window_size:
        sentence_token_padded = sentence_token + [PAD_token]* (window_size - len(sentence_token))     
    return sentence_token_padded
    

def tokenize( voc, txt_words, window_size ):
    '''loop through input_words or label_words to tokenize every sentences'''
    txt_tokenized = [ ]
    for i in range(len(txt_words)):
        txt_tokenized.append( zeroPadding( indexesFromSentence(voc, txt_words[i]), window_size) )
        
    return txt_tokenized


        