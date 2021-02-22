import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


def unique_words(lines):
    unique_words = set()
    stop_words = set(stopwords.words('english'))

    for line in lines:
        for word in line:
            if word not in stop_words:
                unique_words.add(word)
    word_vals = set(i for i in range(len(unique_words)))
    word_keys = dict(zip(unique_words, word_vals))
    return word_keys

def word_to_vec(posts):
    stop_words = set(stopwords.words('english'))
    integer_arr = []
    posts = [line.lower().translate(str.maketrans('', '', string.punctuation)).split(' ') for line in posts]
    word_dict = unique_words(posts)
    word_key_map = [[word_dict[word] for word in post if word not in stop_words] for post in posts]
    return word_key_map


def main():
    yelp_labelled = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
    yelp_labelled.columns = ['text', 'label']

    positive_posts = pd.Series.to_numpy(yelp_labelled[yelp_labelled.label == 1]['text'])
    negative_posts = pd.Series.to_numpy(yelp_labelled[yelp_labelled.label == 0]['text'])

    posts = positive_posts + negative_posts

    neg_one_hot = word_to_vec(negative_posts)
    pos_one_hot = word_to_vec(positive_posts)

    one_hot_arr = np.concatenate((np.zeros(len(neg_one_hot)), np.ones(len(pos_one_hot))))

    X_train, X_test, y_train, y_test = train_test_split(np.concatenate((neg_one_hot, pos_one_hot)),
                                                        one_hot_arr,
                                                        test_size=0.33)

    # Padding the data samples to a maximum review length in words
    longest_post = max(len(post) for post in word_to_vec(posts))
    X_train = sequence.pad_sequences(X_train, maxlen=longest_post)
    X_test = sequence.pad_sequences(X_test, maxlen=longest_post)
    print('X_train shape:', X_train.shape, y_train.shape)
    print('X_test shape:', X_test.shape, y_test.shape)


    ##### LSTM
    #todo:
    # input dimension = Size of the vocabulary, i.e. maximum integer index + 1.
    # adjust embedding size, number of epochs, plot val accuracy vs actual accuracy
    # try other loss functions
    vocab_size = max([max(arr) for arr in word_to_vec(posts)])
    embedding_size = 128
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=longest_post))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    batch_size = 64
    num_epochs = 10
    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    main()
