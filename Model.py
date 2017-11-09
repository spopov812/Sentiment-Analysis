from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Flatten
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def build_model():

    model = Sequential()

    model.add(Embedding(5000, 32, input_length=200, batch_size=32))

    model.add(LSTM(32, batch_size=1, return_sequences=True, activation="relu"))
    model.add(Dropout(.5))

    model.add(LSTM(32, return_sequences=True, activation="relu"))
    model.add(Dropout(.5))

    model.add(LSTM(32, return_sequences=True, activation="relu"))
    model.add(Dropout(.5))

    model.add(Flatten())

    model.add(Dense(4, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["binary_accuracy", 'categorical_crossentropy']
    )

    return model


def train_model():

    vocab_size = 5000
    max_review_length = 200

    data = pd.read_csv("CleanedReviews.csv")

    X_temp = data['Text']
    Y_temp = data['Score']

    X_data = []
    Y_data = []

    for sentence in X_temp:

            X_data.append(sentence)

    X_data = [one_hot(word, vocab_size) for word in X_data]
    X_data = pad_sequences(X_data, maxlen=max_review_length)

    for val in Y_temp:

        if val == 1:
            Y_data.append([1, 0, 0, 0])
        elif val == 2:
            Y_data.append([0, 1, 0, 0])
        elif val == 4:
            Y_data.append([0, 0, 1, 0])
        else:
            Y_data.append([0, 0, 0, 1])

    Y_data = np.array(Y_data)
    X_data = np.array(X_data)

    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

    model = build_model()

    # y_train = np.array(y_train)

    history = model.fit(x_train, y_train, epochs=2, batch_size=32)

    plt.plot(history.history["binary_accuracy"])
    plt.plot(history.history["categorical_crossentropy"])

    plt.show()

    model.save("1Epoch.h5")