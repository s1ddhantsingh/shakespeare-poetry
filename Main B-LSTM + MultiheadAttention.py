import tensorflow as tf
import json
import numpy as np
import os

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Layer


class MultiHeadAttention(Layer):
    def __init__(self, num_heads, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


# Path to the directory containing individual song files
directory = 'responses/txt'
temperature = 0.80

# Read and process each song file
songs = []
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as file:
        songs.append(file.read())

# Tokenize the songs into words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(songs)
sequences = tokenizer.texts_to_sequences(songs)

# Prepare the training data
vocab_size = len(tokenizer.word_index) + 1
sequence_length = 10  # Adjust this value as needed

input_sequences = []
output_sequences = []
for sequence in sequences:
    for i in range(sequence_length, len(sequence)):
        input_sequences.append(sequence[i-sequence_length:i])
        output_sequences.append(sequence[i])

X = np.array(input_sequences)
y = np.eye(vocab_size)[np.array(output_sequences)]

# Create the LSTM model
# Create the LSTM model
num_heads = 8  # Number of attention heads
d_model = 128  # Dimension of the model

input_layer = Input(shape=(sequence_length,))
embedding_layer = Embedding(
    vocab_size, d_model, input_length=sequence_length)(input_layer)
lstm_layer = Bidirectional(
    LSTM(d_model, return_sequences=True))(embedding_layer)
dropout_layer = Dropout(0.35)(lstm_layer)

# Create separate layers for 'v', 'k', and 'q'
v = dropout_layer
k = dropout_layer
q = dropout_layer

attention_layer = MultiHeadAttention(num_heads, d_model)(v, k, q)
lstm_layer_2 = Bidirectional(LSTM(d_model))(attention_layer)
dropout_layer_2 = Dropout(0.20)(lstm_layer_2)
output_layer = Dense(vocab_size, activation='softmax')(dropout_layer_2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam')


def generate_text(epoch, _):
    start_sequence = np.random.randint(0, len(X))
    input_sequence = X[start_sequence]
    generated_text = ''

    for _ in range(100):  # Generate 100 words
        x_pred = np.expand_dims(input_sequence, axis=0)
        predictions = model.predict(x_pred, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Apply top-k sampling
        top_k = 10  # Adjust as needed
        top_k_indices = np.argpartition(predictions, -top_k)[-top_k:]
        top_k_probs = predictions[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)  # Re-normalize

        next_index = np.random.choice(top_k_indices, p=top_k_probs)
        next_word = tokenizer.index_word[next_index]

        generated_text += next_word + ' '
        input_sequence = np.append(
            input_sequence[1:], np.expand_dims(next_index, axis=0))

    print(generated_text)


# Define the callback for text generation during training
generate_text_callback = LambdaCallback(on_epoch_end=generate_text)

# Train the LSTM model
model.fit(X, y, epochs=200, batch_size=128, callbacks=[generate_text_callback])

# Save the trained model
model.save('song_generator_model_BRNN_and_multihead_ATTENTION.h5')

# Save the tokenizer as JSON
with open('tokenizer.json', 'w') as tokenizer_file:
    json.dump(tokenizer.to_json(), tokenizer_file)
