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

counts = count_vect.transform(fixed_text)

# now we try to make predictions
# all the different optionsin naive_bayes are fine
# what this is doing is it's trying to learn if it's positive, negative or neutral
# fixed target = positive, negative, neutral
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts, fixed_target)

<<<<<<< HEAD
print nb.predict(count_vect.transform(["I hate my iphone!!!"]))
=======
print nb.predict(count_vect.transform(["iphone!!!"]))
>>>>>>> 615991c8d373c7e7c162d071b0dec451bbb02fea
#print nb.predict(count_vect.transform(["iphone cost too much!!!"]))
