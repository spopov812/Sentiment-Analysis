from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def build_model():

    model = Sequential()

    model.add(Embedding(5000, 32, input_shape=600))

    model.add(LSTM(32))
    model.add(Dropout(.5))

    model.add(LSTM(32))
    model.add(Dropout(.5))

    model.add(LSTM(32))
    model.add(Dropout(.5))

    model.add(Dense(4, activation="softmax"))

    model.compile(

        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["binary_accuracy", 'categorical_crossentropy']
    )

    return model


def train_model():

    max_size = 5000

    data = pd.read_csv("CleanedReviews.csv")

    X_data = data['Test']
    Y_temp = data['Score']
    Y_data = []

    X_data = sequence.pad_sequences(X_data, maxlen=max_size)

    for val in Y_temp:

        if val == 1:
            Y_data.append([1, 0, 0, 0])
        elif val == 2:
            Y_data.append([0, 1, 0, 0])
        elif val == 4:
            Y_data.append([0, 0, 1, 0])
        else:
            Y_data.append([0, 0, 0, 1])

    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3)

    model = build_model()

    history = model.fit(x_train, y_train, batch_size=32, epochs=1)

    plt.plot(history.history["binary_accuracy"])
    plt.plot(history.history["categorical_crossentropy"])

    plt.show()
