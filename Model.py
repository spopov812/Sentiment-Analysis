from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding


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
