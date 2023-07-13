from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import io
import numpy as np

print('hello')


def adjust_sequence_length(sequence, desired_length):
    if len(sequence) > desired_length:
        # If the sequence is too long, truncate it
        return sequence[:desired_length]
    elif len(sequence) < desired_length:
        # If the sequence is too short, pad it with spaces
        return sequence.ljust(desired_length)
    else:
        # If the sequence is already the right length, return it as is
        return sequence


sequence = ''.join([chr(i) for i in range(97, 123)]) + " "
sequence_length = len(sequence)
num_unique_chars = len(set(sequence))
char_to_int = {ch: i for i, ch in enumerate(sequence)}
int_to_char = {i: ch for i, ch in enumerate(sequence)}

model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, num_unique_chars)))
model.add(Dense(num_unique_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

generated_sequence = adjust_sequence_length(
    "the quick brown fox", sequence_length)

for i in range(10000):  # Generate 100 characters
    x_pred = np.zeros((1, sequence_length, num_unique_chars))
    for t, char in enumerate(generated_sequence):
        x_pred[0, t, char_to_int[char]] = 1.

    predictions = model.predict(x_pred, verbose=0)[0]
    next_index = np.argmax(predictions)
    next_char = int_to_char[next_index]

    generated_sequence = generated_sequence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
