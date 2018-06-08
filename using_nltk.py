#!/usr/bin/env python
"""
    File name: 
    Description:
    Author: Rishabh Gupta
    Date created:
    Date last modified:
    Python Version: 2.7
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import nltk
# %matplotlib inline
from subprocess import check_output


data = pd.read_csv('input/Sentiment.csv')
data = data[['text','sentiment']]

train, test = train_test_split(data, test_size=0.1)

train_pos = train[ train['sentiment'] == 'Positive']    # Filter positive data
train_pos = train_pos['text']   # only text
train_neg = train[ train['sentiment'] == 'Negative']    # Filter negative data
train_neg = train_neg['text']

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Positive words")
# wordcloud_draw(train_pos,'white')
print("Negative words")
# wordcloud_draw(train_neg)

# print "hi"
tweets = []
# print STOPWORDS
# stopwords_set = set(STOPWORDS.words("english"))
stopwords_set = STOPWORDS
# print stopwords_set

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[ test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'Negative']
test_neg = test_neg['text']


def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all
# get words in tweets
all_words = get_words_in_tweets(tweets)


# get word feature
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist) # This will return dictionary with frequency of each word
    features = wordlist.keys() # extract
    return features
    print "hi"

# subset = all_words[:200]
# w_features = get_word_features(subset)
w_features = get_word_features(all_words)
print "hi"


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# wordcloud_draw(w_features)
training_set = nltk.classify.apply_features(extract_features,tweets) #  returns an object that acts like a list but does not store all the feature sets in memory:
classifier = nltk.NaiveBayesClassifier.train(training_set)
print "hi"
neg_cnt = 0
pos_cnt = 0
for obj in test_neg:
    res = classifier.classify(extract_features(obj.split()))
    if (res == 'Negative'):
        neg_cnt = neg_cnt + 1
for obj in test_pos:
    res = classifier.classify(extract_features(obj.split()))
    if (res == 'Positive'):
        pos_cnt = pos_cnt + 1

print('[Negative]: %s/%s ' % (len(test_neg), neg_cnt))
print('[Positive]: %s/%s ' % (len(test_pos), pos_cnt))

if __name__ == "__main__":
    pass
    # print extract_features("movie is good and entertaining ")