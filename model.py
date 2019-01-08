import tensorflow as tf

class Model:
    """
        This is a seq2seq model.
    """

    def __init__(self, MAX_LEN, vocab_size):
        """
        Initialize a Seq2Seq Model with the given data.

        :param window_size: max len of French padded sentence, integer
        :param vocab_size: Vocab size, integer
        """

        # Initialize Placeholders
        self.vocab_size = vocab_size
        self.window_size = MAX_LEN + 2

        self.encoder_input = tf.placeholder(tf.int32, shape=[None, self.window_size], name='input')
        self.encoder_input_length = tf.placeholder(tf.int32, shape=[None], name='input_length')

        self.decoder_input = tf.placeholder(tf.int32, shape=[None, (self.window_size - 1)], name='decoder_input')
        self.decoder_input_length = tf.placeholder(tf.int32, shape=[None], name='label_length')
        self.decoder_labels = tf.placeholder(tf.int32, shape=[None, (self.window_size - 1)], name='labels')
        
        self.keep_prob = tf.placeholder(tf.float32)

        # Please leave these variables
        self.logits = self.forward_pass()
        self.loss = self.loss_function()
        self.train = self.back_propagation()
        self.accWords = self.accuract_words()

    def forward_pass(self):
        """
        Calculates the logits

        :return: A tensor of size [batch_size, english_window_size, english_vocab_size]
        
        """
        embedSz = 300
        rnnSz = 150
        
        with tf.variable_scope("enc"):
            EI = tf.Variable(tf.random_normal([self.vocab_size, embedSz], stddev = .1) )
            embs = tf.nn.embedding_lookup(EI, self.encoder_input)
            embs = tf.nn.dropout(embs, self.keep_prob)
            cell = tf.contrib.rnn.GRUCell(rnnSz)
            initState = cell.zero_state(tf.shape(self.encoder_input)[0], tf.float32)
            encOut, encState = tf.nn.dynamic_rnn(cell, embs, self.encoder_input_length, initState)
            #print(encOut.shape)
            wAT = tf.Variable(tf.random_normal([self.window_size, self.window_size -1], stddev = .1) )
            encOut = tf.tensordot(encOut, wAT, [[1], [0]])
            encOut = tf.transpose(encOut, [0,2,1])
            #print(encOut.shape)
            
        with tf.variable_scope("dec"):
            EO = tf.Variable(tf.random_normal([self.vocab_size, embedSz], stddev = .1))
            embs = tf.nn.embedding_lookup(EO, self.decoder_input)
            embs = tf.nn.dropout(embs, self.keep_prob)
            embb = tf.concat([embs, encOut], axis = 2)
            #print(embb.shape)
            cell = tf.contrib.rnn.GRUCell(rnnSz)
            decOut, _ = tf.nn.dynamic_rnn(cell, embb, initial_state = encState)
            #print(decOut.shape)
            
        W = tf.Variable(tf.random_normal([rnnSz, self.vocab_size], stddev = .1) )
        b = tf.Variable(tf.random_normal([self.vocab_size], stddev = .1))
        logits = tf.tensordot(decOut, W, axes = [[2], [0]]) + b
        #print(logits.shape)
        
        return(logits)
            

    def loss_function(self):
        """
        Calculates the model cross-entropy loss after one forward pass

        :return: the loss of the model as a tensor (averaged over batch)
        """
        mask = tf.cast(tf.sequence_mask( tf.add(self.decoder_input_length, 1) , self.window_size -1), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.decoder_labels, mask)
                                                
        return(loss)


    def back_propagation(self):
        """
        Adds optimizer to computation graph

        :return: optimizer
        """
        
        train_op = tf.train.AdamOptimizer(learning_rate= 10**(-3)).minimize(self.loss)
        return(train_op)
        
    def accuract_words(self):
        pred_labels = tf.cast(tf.argmax(self.logits, axis = 2), tf.int32)
        
        weights = tf.cast(tf.sequence_mask( tf.add(self.decoder_input_length, 1) , self.window_size - 1), tf.float32)
        equality = tf.cast(tf.equal(pred_labels, self.decoder_labels), tf.float32)
        bool_mul_weights = tf.multiply(weights, equality)
        acc_words = tf.reduce_sum(bool_mul_weights)
        
        #return(bool_mul_weights)
        return(acc_words)
