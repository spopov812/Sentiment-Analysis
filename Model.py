from keras import Sequential
from keras.layers import Dense, Recurrent, LSTM, Dropout, Embedding


def build_model():

    model = Sequential()

    model.add(Embedding(1000, 64, input_shape=100))

    model.add(LSTM(32))
    model.add(Dropout(.4))

    model.add(LSTM(32))
    model.add(Dropout(.4))

    model.add(LSTM(32))
    model.add(Dropout(.4))

    model.add(Dense(4, activation="softmax"))

    model.compile(

        loss="mse",
        optimizer="adam",
        metrics=["binary_accuracy"]


    )