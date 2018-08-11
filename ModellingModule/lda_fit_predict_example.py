# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:54:52 2018

@author: c14238a
"""

# The LinkedIn article actually explains things! At least some of them... in any case
# have a look.

# derived from http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
# explanations are located there : https://www.linkedin.com/pulse/dissociating-training-predicting-latent-dirichlet-lucien-tardres

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

n_features = 50
n_topics = 2

# Training dataset
data_samples = ["I like to eat broccoli and bananas.",
                "I ate a banana and spinach smoothie for breakfast.",
                "Chinchillas and kittens are cute.",
                "My sister adopted a kitten yesterday.",
                "Look at this cute hamster munching on a piece of broccoli."
               ]
# extract fetures and vectorize dataset
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)

#save features
dic = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

# train LDA
p1 = lda.fit(tf)

# Save all data necessary for later prediction
model = (dic,lda.components_,lda.exp_dirichlet_component_,lda.doc_topic_prior_)

with open('outfile', 'wb') as fp:
    pickle.dump(model, fp)








# derived from http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
# explanations are located there : https://www.linkedin.com/pulse/dissociating-training-predicting-latent-dirichlet-lucien-tardres

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle

# create a blank model
lda = LatentDirichletAllocation()

# load parameters from file
with open ('outfile', 'rb') as fd:
    (features,lda.components_,lda.exp_dirichlet_component_,lda.doc_topic_prior_) = pickle.load(fd)

# the dataset to predict on (first two samples were also in the training set so one can compare)
data_samples = ["I like to eat broccoli and bananas.",
                "I ate a banana and spinach smoothie for breakfast.",
                "kittens and dogs are boring"
               ]
# Vectorize the training set using the model features as vocabulary
tf_vectorizer = CountVectorizer(vocabulary=features)
tf = tf_vectorizer.fit_transform(data_samples)

# transform method returns a matrix with one line per document, columns being topics weight
predict = lda.transform(tf)
print(predict)

# OK, I have to understand the matrix, and then to actually extract the topic keywords.