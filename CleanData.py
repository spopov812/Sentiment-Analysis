import pandas as pd


df = pd.read_csv('Reviews.csv')

df = df[df.Score != 3]

df = df[df.HelpfulnessNumerator / df.HelpfulnessDenominator >= .5]

df = df.drop(

        ['Id', 'ProductId', 'UserId', 'ProfileName', 'Time',
         'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Summary'],

             axis=1

        )

df.to_csv('CleanedReviews.csv')
