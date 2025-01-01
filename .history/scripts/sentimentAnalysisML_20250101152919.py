# importing libraries and setting variables
import pickle
import csv
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# plot the model performance graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history["val_" + string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


def create_save_sentiment_model():
  vocab_size = 3500 # from 10000
  embedding_dim = 16
  max_length = 100
  training_size = int(1967 * 0.95)
  testing = True

  # data collection and processing
  sentences = []
  labels = []

  with open('sentimentData.csv', mode='r', encoding='utf-8', errors="ignore") as file:
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

  # Sequencing: Sequences of numbers (words) to represent sentences
  # - 2D Array --> Each subarray represents a sentence and each item in the subarray is the number representing the word
  # Pad (with 0's) to ensure all sentences are the same length
  training_sequences = tokenizer.texts_to_sequences(training_sentences)
  training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post', truncating='post') 
  testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
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

  # train the model and plot the accuracy/loss graphs
  history = model.fit(training_padded, training_labels, epochs=200, validation_data=(testing_padded, testing_labels), verbose=2)

  if testing:
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

  # download/save the sentiment analysis model and tokenizer
  model.save('tf_model.keras')

  with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sentiment_from_sentences(sentences):

  testing = False
  # use the model to predict sentence sentiment values (closer to 0 means negative sentiment AND closer to 1 means positive sentiment)
  #   EXAMPLE INPUT: sentences = ["the company had sales increase by 10%", "falling", "Analysts predict a decline in AMD's stock price over the next year due to increased competition, market uncertainties, and potential slowdowns in consumer demand for semiconductor products."]
  
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  
  sequences = tokenizer.texts_to_sequences(sentences)
  padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

  loaded_model = tf.keras.models.load_model('tf_model.keras')
  predictions = loaded_model.predict(padded)

  if testing:
    print(sentences)
    print(predictions)

  the_sentiment = predictions[0][0] # sentiment score of first string

  the_sentiment = round(the_sentiment) # for dis

  return the_sentiment

  # return sentiment score of first string
  if predictions[0][0] >= 0.5:
    return 1 # positive sentiment
  return 0 # negative sentiment
   



# print("\n\n\n\n\n\nLOADING MODEL BELOW\n\n\n\n\n")
# sentiment_from_sentences(["the company had sales increase by 10%", "falling", "Analysts predict a decline in AMD's stock price over the next year due to increased competition, market uncertainties, and potential slowdowns in consumer demand for semiconductor products."])
# sentiment_from_sentences(["the company had sales decrease by 10%. falling. Analysts predict a decline in AMD's stock price over the next year due to increased competition, market uncertainties, and potential slowdowns in consumer demand for semiconductor products."])