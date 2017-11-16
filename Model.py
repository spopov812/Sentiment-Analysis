from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Flatten
from keras.layers import LeakyReLU
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def build_model():

    model = Sequential()

    model.add(Embedding(8000, 64, input_length=500))

    model.add(LSTM(32, return_sequences=True))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.5))

    model.add(LSTM(32, return_sequences=True))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dropout(.5))
    '''
    model.add(LSTM(32, return_sequences=True, activation="LeakyReLU"))
    model.add(Dropout(.5))
    '''
    model.add(Flatten())

    model.add(Dense(2, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer=adam(lr=.001),
        metrics=["binary_accuracy", 'categorical_crossentropy', 'categorical_accuracy']
    )

    return model


def train_model(test_size):

    vocab_size = 8000
    max_review_length = 500

    data = pd.read_csv("CleanedReviews.csv")

    X_temp = data['Text']
    Y_temp = data['Score']

    X_data = []
    Y_data = []

    for sentence in X_temp:

            X_data.append(sentence.lower())

    X_data = [one_hot(word, vocab_size) for word in X_data]
    X_data = pad_sequences(X_data, maxlen=max_review_length)

    for val in Y_temp:

        if val == 1 or val == 2:
            Y_data.append([1, 0])

        elif val == 4 or val == 5:
            Y_data.append([0, 1])

    Y_data = np.array(Y_data)
    X_data = np.array(X_data)

    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=test_size)

    x_test.tofile("XTesting_Data")
    y_test.tofile("YTesting_Data")

    model = build_model()

    callbacks = []

    callbacks.append(TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True))

    callbacks.append(ModelCheckpoint("Epoch-{epoch:02d}-{categorical_crossentropy:.4f}.h5",
                                     monitor='categorical_crossentropy', verbose=0,
                                     save_best_only=True, save_weights_only=False, mode='auto', period=1))

    # training and saving model
    model.fit(x_train, y_train, epochs=4, callbacks=callbacks)

