import pandas as pd
import numpy as np


df = pd.read_csv('tweets.csv')
target = df['is_there_an_emotion_directed_at_a_brand_or_product']
text = df['tweet_text']

fixed_text = text[pd.notnull(text)]
fixed_target = target[pd.notnull(text)]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

p = Pipeline(steps=[('counts', CountVectorizer(ngram_range=(1, 2))),
                ('multinomialnb', MultinomialNB())])

p.fit(fixed_text, fixed_target)
<<<<<<< HEAD

# print the index of the count for the bigram 'garage sale'
#print p.named_steps['counts'].vocabulary_.get(u'garage sale')

# number of bigrams
=======
#print p.named_steps['counts'].vocabulary_.get(u'garage sale')
>>>>>>> 615991c8d373c7e7c162d071b0dec451bbb02fea
print len(p.named_steps['counts'].vocabulary_)
