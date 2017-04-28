import pandas as pd
import numpy as np

df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']
print text

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

# Data cleaning - this deals with null lines of text
fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

# this counts the number of times it's seen a word in the text
count_vect.fit(fixed_text)

# prints number of time saw '3g' in column of tweets
print count_vect.vocabulary_.get(u'the')

# Lets us change any text we have into a giant array
# It looks at any given tweet and has a giant dict of words and counts how many
# times it has seen any of these words in a tweet_text
print count_vect.transform(["a"])
