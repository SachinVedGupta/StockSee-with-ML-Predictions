# importing libraries and setting variables
import pickle
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# plot the model performance graphs
def plot_graphs(history, string, filename=None, ticker=None):

  the_title = string + f" graph"
  if ticker:
     the_title += f"for {ticker} "
  plt.plot(history.history[string])
  plt.plot(history.history["val_" + string])
  plt.title(the_title)
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])

  if (filename):
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the plot to a file
    print(f"Plot saved as {filename}")
  else:
    plt.show()


def create_save_sentiment_model(): # trains the sentiment analysis ML model and saves it for later use
  vocab_size = 3500 # from 10000
  embedding_dim = 16
  max_length = 100
  training_size = int(1967 * 0.95)
  testing = False

  # data collection and processing
  sentences = []
  labels = []

  with open('./sentiment_storage/sentimentData.csv', mode='r', encoding='utf-8', errors="ignore") as file:
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
  model.save('./sentiment_storage/tf_model.keras')

  with open('./sentiment_storage/tokenizer.pickle', 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



def sentiment_from_sentence(sentence): # uses the already saved sentiment analysis model to get the sentiment score for an input sentence
  sentences = [sentence]

  testing = False
  # use the model to predict sentence sentiment values (closer to 0 means negative sentiment AND closer to 1 means positive sentiment)
  #   EXAMPLE INPUT: sentence = "the company's sales had increased by 10%"
  
  with open('./sentiment_storage/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  
  sequences = tokenizer.texts_to_sequences(sentences)
  padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

  loaded_model = tf.keras.models.load_model('./sentiment_storage/tf_model.keras')
  predictions = loaded_model.predict(padded)

  if testing:
    print(sentences)
    print(predictions)

  the_sentiment = predictions[0][0] # sentiment score of first string
  # the_sentiment = round(the_sentiment) # for discretization of the sentiment score

  return the_sentiment # decimal value between 0 and 1 representing the sentiment analysis score --> closer to 1 is positive sentiment AND closer to 0 is negative sentiment