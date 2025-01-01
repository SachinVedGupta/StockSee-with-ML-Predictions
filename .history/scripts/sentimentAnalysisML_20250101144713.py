# importing libraries and setting variables

import csv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_save_sentiment_model():
    

vocab_size = 3500 # from 10000
embedding_dim = 16
max_length = 100
training_size = int(1967 * 0.95)


# data collection and processing

sentences = []
labels = []

with open('all-data.csv', mode='r', encoding='utf-8', errors="ignore") as file:
    reader = csv.reader(file)
    for row in reader:

        if row[0] == 'negative':
            labels.append(0)
            sentences.append(row[1])
        elif row[0] == 'positive':
            labels.append(1)
            sentences.append(row[1])


training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# Preparing the sentences for the Neural Network

# Tokenizer: Create Tokenizer object --> each word (in input paragraph) is assigned a unique key (hashmap format) --> now words can be represented by numbers
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") # any words not initially used in fit_on_texts is assigned to the oov token (assigned number 1)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# Sequencing: Sequences of numbers (words) to represent sentences
# - 2D Array --> Each subarray represents a sentence and each item in the subarray is the number representing the word
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post') 

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# Pad (with 0's) to ensure all sentences are the same length
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post', truncating='post')

# convert to numpy array for proper format
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# create the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train the model

history = model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), verbose=2)

# plot the model performance graphs

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history["val_" + string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# test out the model with input sentences (closer to 0 means negative sentiment AND closer to 1 means positive sentiment)

sentence = ["the company had sales increase by 10%", "falling", "Analysts predict a decline in AMD's stock price over the next year due to increased competition, market uncertainties, and potential slowdowns in consumer demand for semiconductor products."]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
print(model.predict(padded))

model.save('tf_model.h5')
loaded_model = tf.keras.models.load_model('tf_model.h5')
loaded_model.predict(padded)