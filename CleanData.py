import pandas as pd


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

# strips away some artifact in the provided data
df['Text'] = df['Text'].replace('<br />', ' ')

# saves data frame to .csv
df.to_csv('CleanedReviews.csv')
