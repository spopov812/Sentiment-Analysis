from Model import train_model
from CleanData import clean_data
from keras.models import load_model
import numpy as np
import sys
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.ERROR)


def test():

	print("\nLoading testing data...")

	x_data = np.array(np.load("XTesting_Data.npy"))
	y_data = np.array(np.load("YTesting_Data.npy"))

	print("Testing...\n")

	scores = model.evaluate(x_data, y_data, verbose=0)

	print("Accuracy: %.2f%%" % (scores[1]*100))


# Valid arg count
if len(sys.argv) < 2:
	print("\nInvalid args.\n")


# arg parsing
if "clean" in sys.argv:
	clean_data()

if "train" in sys.argv:

	to_test = input("\n\nWhat percent of the data should be used in training (as decimal)\n")
	train_model(1 - float(to_test))

if "test" in sys.argv:

	model_name = input("\n\nPlease provide name of model to evaluate- \n")

	try:
		model = load_model(model_name)
	except OSError:
		try:
			model = load_model(model_name + ".h5")
		except OSError:
			print("File not found")
			exit(0)

	test()
