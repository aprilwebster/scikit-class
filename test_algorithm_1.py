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

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(counts, fixed_target)

# this makes a prediction on every single tweet in the array, and want
# count how many times there's a mismatch between the prediction and the actual
#
predictions = nb.predict(counts)
<<<<<<< HEAD
correct = sum(predictions == fixed_target)
incorrect = sum(predictions != fixed_target)

# what's wrong with the accuracy calculation?
# note, the algorith accuracy is about 70% - it's 
# we tested the algorithm on the data we trained the algorithm on
# this is a really bad thing as we need to test how well the alg generalizes
# so instead we need to do a test/train split
# in machine learning it's common to do 70% for training, and 30% for testing
accuracy = correct/(correct + incorrect)
print 'accuracy = ' + str(accuracy * 100)
=======
correct= sum(predictions == fixed_target)
incorrect= sum(predictions != fixed_target)
acc = correct/(len(fixed_target)+0.)
print(acc)
>>>>>>> 615991c8d373c7e7c162d071b0dec451bbb02fea
