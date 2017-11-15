from Model import train_model
from CleanData import clean_data
import sys

if len(sys.argv) < 2:
    print("Invalid args.")

if "clean" in sys.argv:
    clean_data()

if "train" in sys.argv:
    train_model()

if "test" in sys.argv:
    print("Todo")

if "run" in sys.argv:
    print("Todo")
