from Model import train_model
from CleanData import clean_data
from keras.models import load_model
import numpy as np
import sys


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

def test_user_review(review, model):

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

    model_name = input("\n\nPlease provide name of model to evaluate\n")
    model = load_model(model_name)

    print("Type q to exit.\n")

    while True:
        review = input("\n\nWhat is the review?\n")

        if review == 'q':
            break


