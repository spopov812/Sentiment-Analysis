import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

#nltk.download()

def clean_data():
    df = pd.read_csv('Reviews.csv')

    # removes products with a medium score
    df = df[df.Score != 3]

    # removes reviews that the majority of people don't think is helpful
    df = df[df.HelpfulnessNumerator / df.HelpfulnessDenominator >= .5]

    # removes unnecessary columns
    df = df.drop(

            ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time',
             'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Summary'],

            axis=1

            )

    # saves data frame to .csv
    df.to_csv('CleanedReviews.csv')
