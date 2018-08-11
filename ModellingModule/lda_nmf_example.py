# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:11:48 2018

@author: c14238a
"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

#print("Loading dataset...")
#t0 = time()
#dataset = fetch_20newsgroups(shuffle=True, random_state=1,
#                             remove=('headers', 'footers', 'quotes'))
#data_samples = dataset.data[:n_samples]
#print("done in %0.3fs." % (time() - t0))
#
## Use tf-idf features for NMF.
#print("Extracting tf-idf features for NMF...")
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
#                                   max_features=n_features,
#                                   stop_words='english')
#t0 = time()
#tfidf = tfidf_vectorizer.fit_transform(data_samples)
#print("done in %0.3fs." % (time() - t0))
#
## Use tf (raw term count) features for LDA.
#print("Extracting tf features for LDA...")
#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
#                                max_features=n_features,
#                                stop_words='english')
#t0 = time()
#tf = tf_vectorizer.fit_transform(data_samples)
#print("done in %0.3fs." % (time() - t0))
#print()
#
## Fit the NMF model
#print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
#      "n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
#t0 = time()
#nmf = NMF(n_components=n_components, random_state=1,
#          alpha=.1, l1_ratio=.5).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))
#
#print("\nTopics in NMF model (Frobenius norm):")
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)
#
## Fit the NMF model
#print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
#      "tf-idf features, n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
#t0 = time()
#nmf = NMF(n_components=n_components, random_state=1,
#          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
#          l1_ratio=.5).fit(tfidf)
#print("done in %0.3fs." % (time() - t0))
#
#print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(nmf, tfidf_feature_names, n_top_words)
#
#print("Fitting LDA models with tf features, "
#      "n_samples=%d and n_features=%d..."
#      % (n_samples, n_features))
#lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
#                                learning_method='online',
#                                learning_offset=50.,
#                                random_state=0)
#t0 = time()
#lda.fit(tf)
#print("done in %0.3fs." % (time() - t0))
#
#print("\nTopics in LDA model:")
#tf_feature_names = tf_vectorizer.get_feature_names()
#print_top_words(lda, tf_feature_names, n_top_words)





from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
#documents = a.cleaned_documents

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

# Display the most probable topic for the trained documents
doc_topic = lda.transform(tf)
# the rows of doc_topic are the documents, the columns are the topic probabilities
# loop over each document and for each document find the highest value (i.e. topic)
for n in range(doc_topic.shape[0]):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {}    topic: {}\n".format(n,topic_most_pr))
    
    
### test try new data
with open('D:\Книги\Мои\Python\Articles\Articles\\sampleCH.txt', 'r') as f:
    new_doc = f.read()
new_doc = [new_doc]
    
# NMF try (note min_df and max_df)
_tfidf_vectorizer = TfidfVectorizer(max_df = 1, min_df = 1, max_features = no_features,
                                   stop_words = 'english')
_tfidf = tfidf_vectorizer.transform(new_doc)
_tfidf_feature_names = tfidf_vectorizer.get_feature_names()

test = nmf.transform(_tfidf)
test = test[0]
import numpy as np
some = np.argsort(test)[-10:][::-1]
for topic_idx in some:
    topic = nmf.components_[topic_idx]
    print("Topic number: ", topic_idx)
    print("Topic score: ", test[topic_idx] / sum(test))
    for k in topic.argsort()[-15:][::1]:
        print("Word: ", _tfidf_feature_names[k], " Score: ", topic[k] / sum(topic))


# LDA try
_tf_vectorizer = CountVectorizer(max_df = 1, min_df = 1, max_features = no_features, stop_words = 'english')
_tf = tf_vectorizer.transform(new_doc)
_tf_feature_names = tf_vectorizer.get_feature_names()

test_ = lda.transform(_tf)
for n in range(test_.shape[0]):
    topic_most_pr = test_[n].argmax()
    print("doc: {}    topic: {}\n".format(n,topic_most_pr))

some = np.argsort(test_)[-10:][::-1]
for topic_idx in some:
    topic = lda.components_[topic_idx]
    print("Topic number: ", topic_idx)
    print("Topic score: ", test[topic_idx] / sum(test))
    for k in topic.argsort()[-15:][::1]:
        print("Word: ", _tfidf_feature_names[k], " Score: ", topic[k] / sum(topic))

# Is it correct to use the trained vectorizer's feature names?


## Transform new_feedback to NMF space
#nmf_new_feedback = clf.transform(vectorizer.transform(new_feedback))
## top 10 topics of the new feedback
#compo = np.argsort(nmf_new_feedback)[-10:][::-1]
## Connection between indices and words of tfidf
#feature_names = vectorizer.get_feature_names()
#for topic_idx in compo:
#    # Current nmf topic
#    topic = nmf.components_[topic_idx]
#    print("Topic number: ", topic_idx)
#    print("Topic score: ", nmf_new_feedback[topic_idx]/sum(nmf_new_feedback))
#    # Top 15 words of current topic
#    for k in topic.argsort()[-15:][::-1]:
#        print("Word: ",feature_names[k], "  Score: ", topic[k]/ sum(topic))
        
        
# =============================================================================
    
#The call to transform on a LatentDirichletAllocation model returns an unnormalized document topic distribution. To get proper probabilities, you can simply normalize the result. Here is an example:
#https://stackoverflow.com/questions/40597075/python-sklearn-latent-dirichlet-allocation-transform-v-fittransform?rq=1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# grab a sample data set
dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
train,test = dataset.data[:100], dataset.data[100:200]

# vectorizer the features
tf_vectorizer = TfidfVectorizer(max_features=25)
X_train = tf_vectorizer.fit_transform(train)

# train the model
lda = LatentDirichletAllocation(n_topics=5)
lda.fit(X_train)

# predict topics for test data
# unnormalized doc-topic distribution
X_test = tf_vectorizer.transform(test)
doc_topic_dist_unnormalized = np.matrix(lda.transform(X_test))

# normalize the distribution (only needed if you want to work with the probabilities)
doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)

#To find the top ranking topic you can do something like:

doc_topic_dist.argmax(axis=1)


