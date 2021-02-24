import pandas as pd
import numpy as np
import string
import nltk
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten

nltk.download('stopwords')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from keras.preprocessing.text import Tokenizer


def char_tokenizer(lines):
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    lines_low = [[l.lower() for l in lines]]
    tk.fit_on_texts(lines_low)

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    # Use char_dict to replace the tk.word_index
    tk.word_index = char_dict.copy()
    # Add 'UNK' to the vocabulary
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
    return tk


def embed_weights(tk):
    vocab_size = len(tk.word_index)
    embedding_weights = []
    embedding_weights.append(np.zeros(vocab_size))

    for char,i in tk.word_index.items():
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)

    return np.array(embedding_weights)


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
    posts = [line.lower().translate(str.maketrans('', '', string.punctuation)).split(' ') for line in posts]
    word_dict = unique_words(posts)
    word_key_map = [[word_dict[word] for word in post if word not in stop_words] for post in posts]
    return word_key_map


def word_level(p_posts,n_posts):
    neg_encoded = word_to_vec(n_posts)
    pos_encoded = word_to_vec(p_posts)
    posts = n_posts+p_posts
    word_vecs = np.concatenate((neg_encoded, pos_encoded))

    # Padding the data samples to a maximum review length in words
    longest_post = max(len(post) for post in word_to_vec(posts))
    padded_word_vecs = sequence.pad_sequences(word_vecs, maxlen=longest_post)

    # re-create labels, concatenate with data, and set up test/train split
    labels = np.concatenate((np.zeros(len(neg_encoded)), np.ones(len(pos_encoded))))
    X_train, X_test, y_train, y_test = train_test_split(padded_word_vecs, labels, test_size=0.33)

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
    csv_logger = CSVLogger("model_history_log.csv", append=True)

    model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs,callbacks=[csv_logger])
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])


def char_level(p_posts,n_posts):
    posts = p_posts + n_posts
    tk = char_tokenizer(posts)
    vocab_size = len(tk.word_index)
    pos_seqs = tk.texts_to_sequences(p_posts)
    neg_seqs = tk.texts_to_sequences(n_posts)
    longest_post = max([len(x) for x in np.append(neg_seqs, pos_seqs, axis=0)])

    pos_seq_padded = pad_sequences(pos_seqs, maxlen=longest_post, padding='post')
    neg_seq_padded = pad_sequences(neg_seqs, maxlen=longest_post, padding='post')

    pos_data = np.array(pos_seq_padded, dtype='float32')
    neg_data = np.array(neg_seq_padded, dtype='float32')

    padded_char_vecs = np.append(neg_data, pos_data, axis = 0)

    labels = np.concatenate((np.zeros(len(neg_data)), np.ones(len(pos_data))))
    X_train, X_test, y_train, y_test = train_test_split(padded_char_vecs, labels, test_size=0.33)

    print('X_train shape:', X_train.shape, y_train.shape)
    print('X_test shape:', X_test.shape, y_test.shape)

    embedding_size = 69 #TODO: find where this number comes from
    input_size = 1014
    em_weights = embed_weights(tk)
    embedding_layer = Embedding(vocab_size+1,embedding_size,input_length=input_size,weights=[em_weights])

    cnn_model = Sequential()
    cnn_model.add(embedding_layer)
    cnn_model.add(Conv1D(embedding_size, kernel_size=8, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(10, activation='relu'))
    cnn_model.add(Dense(1, activation='sigmoid'))
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_model.summary())

    num_epochs = 5

    cnn_model.fit(X_train, y_train, epochs=num_epochs, verbose=1)

    scores = cnn_model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])



def main():
    yelp_labelled = pd.read_csv('yelp_labelled.txt', sep='\t', header=None)
    yelp_labelled.columns = ['text', 'label']

    positive_posts = pd.Series.to_numpy(yelp_labelled[yelp_labelled.label == 1]['text'])
    negative_posts = pd.Series.to_numpy(yelp_labelled[yelp_labelled.label == 0]['text'])

    posts = positive_posts + negative_posts

    ##### WORD LEVEL NN #######
    # word_level(positive_posts, negative_posts)

    ##### CHAR LEVEL NN ########
    char_level(positive_posts, negative_posts)






if __name__ == "__main__":
    main()
