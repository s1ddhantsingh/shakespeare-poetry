import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Load the tokenizer from the JSON file

addon = "_2layers_35dropout_200epochs"


tokenizer_json = f'tokenizer{addon}.json'

with open(tokenizer_json, 'r') as tokenizer_file:
    tokenizer_config = json.load(tokenizer_file)

tokenizer = tokenizer_from_json(tokenizer_config)

# Load the trained model
model = load_model(f'song_generator_model{addon}.h5')


def generate_text(model, tokenizer, sequence_length, seed_text, num_words):
    generated_text = seed_text

    for _ in range(num_words):
        # Tokenize the seed text
        tokenized_text = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequence if necessary
        padded_text = pad_sequences(
            [tokenized_text], maxlen=sequence_length, truncating='pre')
        # Predict the next word
        predicted_index = np.argmax(model.predict(padded_text), axis=-1)[0]
        # Convert the predicted index to word
        predicted_word = tokenizer.index_word[int(predicted_index)]
        # Append the predicted word to the generated text
        generated_text += " " + predicted_word
        # Update the seed text for the next iteration
        seed_text += " " + predicted_word
        # Remove the first word from the seed text to maintain the sequence length
        seed_text = " ".join(seed_text.split()[1:])

    return generated_text


seed_text = "Let's jump in and learn about the multi head attention mechanism. "
num_words = 1000
generated_text = generate_text(
    model, tokenizer, sequence_length=10, seed_text=seed_text, num_words=num_words)
print(generated_text)
'''

I'm feeling in the beach that they just like that they just like that they just like that they just like that they just like that they just like that they just like that they just like that they just like that they just like that they ju

In: ""
Out:  here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here here

In: Ok, so it seems that more context means better models
Out: Ok, so it seems that more context means better models i'm not to me the whole way they say about me that's the cruelest promises beautiful of his bed and i got a letter yes i'm doing better i'm killing me and i was your friend is a best guy honey ♪ and this is all all the time was only good of your friends crash in the beach island smiles into something there on you need you know he's for a girl hope so it was a princess this ain't a fairy tale i'm not the one you'll sweep off her feet lead her up the stairwell this ain't
'''
