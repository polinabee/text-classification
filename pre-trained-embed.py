import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential

from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.layers import Embedding



import os
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.keras.optimizer_v1 import Adam


class wordVec:
    def __init__(self, data):
        self.samples = pd.Series.to_numpy(data.text)
        self.labels = pd.Series.to_numpy(data.label)
        self.train_samples = None
        self.val_samples = None
        self.train_labels = None
        self.val_labels = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

    def split(self, n):
        seed = 1337
        rng = np.random.RandomState(seed)
        rng.shuffle(self.samples)
        rng = np.random.RandomState(seed)
        rng.shuffle(self.labels)

        # Extract a training & validation split
        validation_split = n
        num_validation_samples = int(validation_split * len(self.samples))
        self.train_samples = self.samples[:-num_validation_samples]
        self.val_samples = self.samples[-num_validation_samples:]
        self.train_labels = self.labels[:-num_validation_samples]
        self.val_labels = self.labels[-num_validation_samples:]

    def vectorize(self):
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
        text_ds = tf.data.Dataset.from_tensor_slices(self.train_samples).batch(128)
        vectorizer.adapt(text_ds)

        self.train_X = vectorizer(np.array([[s] for s in self.train_samples])).numpy()
        self.test_X = vectorizer(np.array([[s] for s in self.val_samples])).numpy()

        self.test_y = np.array(self.train_labels)
        self.test_y = np.array(self.val_labels)

        return vectorizer.get_vocabulary()


def load_pretrained():
    path_to_glove_file = os.path.join(
        os.path.expanduser("~"), ".keras/datasets/glove.6B.300d.txt"
    )

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def embed_matrix(voc,embed_index):
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc) + 2
    embedding_dim = 300
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embed_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    return embedding_layer

def main():
    yelp_labelled = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
    yelp_labelled.columns = ['text', 'label']

    yelpVec = wordVec(yelp_labelled)
    yelpVec.split(0.3)
    voc = yelpVec.vectorize()
    embed_index = load_pretrained()
    embed_layer = embed_matrix(voc, embed_index)

    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    model = Sequential()

    model.add(embed_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()


    model.fit(yelpVec.train_X,
              yelpVec.train_y,
              batch_size=128,
              epochs=20,
              validation_data=(yelpVec.test_X, yelpVec.test_y))


if __name__ == "__main__":
    main()
