import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_vect.fit(fixed_text)

# counts is a matrix of vectors, one for each tweet,
# that provides the count for each word in the tweet in the CountVectorizer
counts = count_vect.transform(fixed_text)
<<<<<<< HEAD

print(counts[0])

print count_vect.transform(["I love my iphone!!!"])
=======
print(fixed_text[0:2])
print(counts[0:2])
#print(fixed_text[0])
#print count_vect.transform(["cerulean"])
>>>>>>> 615991c8d373c7e7c162d071b0dec451bbb02fea
