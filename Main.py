import tensorflow as tf
import tensorflow_addons as tfa
import json
import numpy as np
import os

from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q = inputs
        k = inputs
        v = inputs
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


# EDIT ADD-ON
addon = "_2layers_35dropout_200epochs"

# Path to the directory containing individual song files
directory = 'responses/txt'
temperature = 0.80

# Read and process each song file
songs = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        song_text = file.read()
        songs.append(song_text)

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
y = np.array(output_sequences)
y = np.eye(vocab_size)[y]

# Create the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.35))
model.add(AttentionLayer())
model.add(LSTM(128))
model.add(Dropout(0.20))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


def generate_text(epoch, _):
    generated_text = ''
    start_sequence = np.random.randint(0, len(X))
    input_sequence = X[start_sequence]

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
model.save(f'song_generator_model{addon}.h5')

# Save the tokenizer as JSON
tokenizer_json = tokenizer.to_json()
with open(f'tokenizer{addon}.json', 'w') as tokenizer_file:
    tokenizer_file.write(json.dumps(tokenizer_json))
