from Model import train_model
from CleanData import clean_data
from keras.models import load_model
import numpy as np
import sys
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


def test(model_name, x_data, y_data):

    total_data_size = len(x_data)
    num_correct = 0

    counter = -1

    for val in x_data:

        counter += 1

        predicted_value = np.argmax(model_name.predict(val), axis=1)

        # correct review sentiment
        if y_data[counter][predicted_value[0]] == 1:
            num_correct += 1

    print("\n\nCorrect sentiment was predicted %d times out of %d test samples (%f percent accuracy).\n" %
        (num_correct, total_data_size, (num_correct/total_data_size)))


if len(sys.argv) < 2:
    print("\nInvalid args.\n")

if "clean" in sys.argv:
    clean_data()

if "train" in sys.argv:

    to_test = input("\n\nWhat percent of the data should be used in training (as decimal)n\n")
    train_model(1 - int(to_test))

if "test" in sys.argv:

    model_name = input("\n\nPlease provide name of model to evaluate\n")
    model = load_model(model_name)

    x_test = input("What is the filename for x testing data?")
    y_test = input("What is the filename for y testing data?")

    test(model, x_test, y_test)

if "run" in sys.argv:

    vocab_size = 8000
    max_review_length = 500

    model_name = input("\n\nPlease provide name of model to evaluate\n")
    model = load_model(model_name)

    print("Type q to exit.\n")

    while True:
        review = input("\n\nWhat is the review?\n")

        if review == 'q':
            break

        review = review.lower()

        review = [one_hot(word, vocab_size) for word in review]
        review = pad_sequences(review, maxlen=max_review_length)

        prediction_decimal = model.predict(review)
        prediction = np.argmax(prediction_decimal)

        if prediction[0] == 0:
            print("%f confident that this review has negative sentiment" % (prediction_decimal[prediction[0]]))
        else:
            print("%f confident that this review has positive sentiment" % (prediction_decimal[prediction[0]]))
