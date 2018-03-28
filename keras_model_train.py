# define a custom softmax function
# usually, the default softmax function in Keras computes the softmax on the last axis of its input. 
# in the following model architecture, we'll be using an input shape of (num_examples, num_chars_in_sentence, vocab_size), and 
# we need to apply softmax to axis 1, as that is the axis that contains the predicted character. This will be clearer in the
# remaining portion of the code.
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
        
# define the layers used in the attention model globally
# Tx is the number of time steps, ie, the number of characters in the input sequence
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)

# calculate the attention function
# s_prev is the hidden state representation of the post-activation LSTM's prvious time step,
# a is the input to the attention model, which is the same as the output from the 1st Bi-Directional LSTM

def one_step_attention(a, s_prev):
    
    # transform s_prev from shape (num_examples, num_hidden_units) to (num_examples, num_charaters, num_hidden_units)
    s_prev = repeator(s_prev)
    
    # concatenate a and s_prev on the last axis, and then feed it into a small fully connected neural net
    concat = concatenator([a, s_prev])
    
    # The output of this layer will be (num_examples, num_characters, 1)
    e = densor(concat)
    
    # now we apply the custom softmax defined earlier on axis 1, that contains the characters in the sequence
    alphas = activator(e)
    
    # make the context vector by obtaining the dot product of alphas and a, as per attention mechanism guidelines
    context = dotor([alphas, a])
    
    return context
    
# n_a is the number of hidden layers in the 1st LSTM, n_s the corresponding value of the post-attention LSTM
n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(vocab_size), activation=softmax)

# start defining the Keras model   
X = Input(shape=(Tx, vocab_size))
s0 = Input(shape=(n_s,), name='s0')
c0 = Input(shape=(n_s,), name='c0')
s = s0
c = c0

# empty list to store the outputs from each time step
outputs = []

# pass the input into a bi directional LSTM, and set return_sequences to True, so that the output from each time step is passed on
# and not only the hidden state from the last time step
a = Bidirectional(LSTM(n_a, return_sequences = True )(X)

# loop through the time steps of the output (Ty is the num of timesteps in the output)
for t in range(Ty):

    # call the attention layer on the outputs from the 1st LSTM
    context = one_step_attention(a, s)
    
    # run the output of the attention layer through the 2nd LSTM, using the hidden states from the previous time steps too, as input
    s, _, c = post_activation_LSTM_cell(context, initial_state = [s,c])
                  
    # pass it through the dense layer with custom softmax activation and 'vocb_size' units, to get the predicted character               
    out = output_layer(s)
                  
    # append the outputs into the 'outputs' list
    outputs.append(out)

# create the model instance
model = Model(inputs = [X, s0, c0], outputs = outputs)

# compile and fit the model
model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

model.fit([all_sentences, s0, c0], relations_in_text_form)
