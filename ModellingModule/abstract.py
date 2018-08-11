# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:57:41 2018

@author: c14238a
"""

#import os
#path = "C://Work//Python//Articles//ModellingModule//"
##path = "D://Книги//Мои//Python//Articles//"
#os.chdir(path)

from abc import ABC, abstractmethod

import nltk
#import gensim
import string

import pandas as pd

''' 
General workflow notes:
    1. Get the raw data. I.e. a list of strings, where each string is an article.
    2. Prepare each article. Each article (a string of characters) should be converted
    to a list, where each element of the list is a separate lemmatized word. No 
    puncutation or capital words.
    3. Do Bag of Words - get the frequencies and prepare plots of the most frequent words:
        a. input - list of documents (strings) - raw or cleaned - depends
        b. output - table with the "n" most frequent words. If there are more than
        one documents - use Tf-Idf
    4. Do Sentiment analysis
        a. input - list of documents (strings) - raw or cleaned - depends
        b. output - frequency table of the emotions
                  - polarity score
    5. Prepare and do an LDA
    6. NMF - and compare with LDA
    

A note: I may consider applying tf-idf to a given sample that the user specifies,
whereas the LDA/NMF/Sentiment analysis be done once when the data is downloaded, say.
So, for example - when a reasonably large sample is collected an LDA is performed
that assigns the keywords for each article in a database field and these keywords 
are the same until another estimation has been carried out. At the same time, if the
user wants to check the keywords for a given time frame, then the tf-idf would
calculate weighted frequencies everytime a sample is pulled from the database.

Or, rather - two sorts of modelling - the first calculates tf-idf, LDA, NMF for the
whole database once. The user, however, also gets these things calculated for the 
sample he requested. So, one set of statistics (LDA results, tf-idf frequencies etc.)
would be calculated for the whole database and stored in there; another set would
be calculated each time a sample is pulled from the database. That way, the user 
would be able to compare the whole database statistics with a particular period's
 statistics.

This would mean that the class below should produce tf-idf, LDA and NMF for a given
set of documents. It will be called programatically when a reestimation is carried
out on the whole database and also it will be called each time when the user sub-samples
the database. Sooo... in simple words, it has to be general enough that it can just
handle whatever data it is supplied with and return the appropriate results.

Also, the sentiment analysis and the named-entity recognition would be calculated
just once, as they are unique and invariant for each article. They would be then
stored into the database.
    
'''

class NLPProcessor(ABC):
    '''
    The main idea of this abstract class is to provide a wrapper (or an abstraction
    if you like) over a certain set of NLP techniques and the data they use.
    So, the implementation of this class should take a list of documents (raw text)
    and produce output using a few NLP techniques. 
    
    The implementation may rely heavily on libraries like NLTK and gensim for all methods, 
    and will provide a wrapper of their functionalities. 
    
    So, the class should take a list of documents as an input, have appropriate
    methods for cleaning the raw texts, and have the desired by the programmer
    methods for outputs. (A note, the input data may vary in type, but in general
    should be some sort of a "list".). The NLP techniques will be applied to each
    document of the list. The outputs should be stored in object attributes that 
    would be accessible via the appropriate Getter methods.
    
    This class concretely will have text-cleaning method(s) and will implement
    the following NLP techniques:
        1. Bag-of-words frequency tables
        2. Sentiment analysis (using the NRC lexicon)
            a. Polarity score
            b. Emotions analysis
        3. Latent Dirichlet Allocation
        4. Non-negative Matrix Factorization
        
    1. The Bag-of-Words should take a cleaned list of documents as an input and 
    produce a table of word freqencies within a document, or if the input contains
    multiple documents - a table of the Tf-Idf weighted most frequent words
    
    2. The sentiment analysis should take a cleaned list of documents as an input
    and produce: 
        - polarity score for all documents (average of the individual documents
    polarity, or rather polarity over the merged documents?)
        - a table of the emotions present in all documents using the NRC lexicon
        
    3. The LDA should extract the hidden (latent) topics for the input documents.
    Synonymically, it should extract the keywords of each topic and produce as
    an output a table, where the columns will be the topics and the contents of
    each column will be the keywords for that topic. Then, it should assign each
    document the most probable topic and its keywords (more details required here)
    
    4. The NMF is sort of optional for now, it should produce similar output as
    the LDA, and can serve as a comparison with the LDA.
    '''
    
    @abstractmethod
    def __init__(self, list_of_documents):
        '''
        The constructor should store the input list of documents in an object
        variable and instantiate the list of the cleaned documents.
        
        Parameters:
            list_of_documents: list; contains one or more strings of text.
        Returns:
            None; instantiates the object
        '''
        
    @abstractmethod
    def clean(self):
        '''
        This method should clean the raw data and store the cleaned data in the
        already initialized object variable.
        The cleaning process contains the following steps:
            1. Convert everything to lower case;
            2. Remove any stopwords;
            3. Remove punctuation;
            4. Tokenize.
            
        Parameters:
            nothing; uses the raw data in self that is provided by __init__
        Returns:
            None; stores the cleaned data as an object variable
        '''
        
    @abstractmethod
    def analyzeFrequencies(self):
        '''
        This method takes the cleaned data and finds the most frequent words in
        it. If there is one document it will use standard frequencies; if there
        are multiple - it will weight using Tf-Idf.
        
        It should be assumed that the documents are more than one. So, the algo-
        rithm would return, say, a dictionary with title:tf-idf most frequent words
        as key:value pairs. This dictionary would be then used as deemed approp-
        riately by the other modules.
        
        Parameters:
            None; uses the cleaned data in self.
        Returns:
            table (DataFrame?) with the most frequent words
        '''
       
    @abstractmethod
    def analyzeSentiment(self):
        '''
        This method takes the cleaed data and analyzes the polarity of the documents
        and the emotions present in them.
        
        Parameters:
            None; uses the cleaned data in self.
        Returns:
            table (DataFrame?) with the most frequent words
        '''
        
    @abstractmethod
    def doNER(self):
        '''
        Performs Named Entity Recognition on a given set of documents.
        '''
    @abstractmethod
    def doLDA(self):
        '''
        This method performs Latent Dirichlet Allocation and assigns the most
        probable topic to each document
        '''
    
    @abstractmethod
    def doNMF(self):
        '''
        This method performs Non-negative Matrix Factorization and assigns the most
        probable topic to each document
        '''
        
        
    @abstractmethod
    def getFrequencies(self, index, n):
        '''
        Returns the (tf-idf) frequencies of the `n` most frequent words in the 
        document of index index
        '''
        
    @abstractmethod
    def getPolarityScore(self, index):
        '''
        Returns the polarity score of the document of index index.
        '''
        
    @abstractmethod
    def getSentiments(self, index):
        '''
        Returns the emotions table of a document with index index
        '''
    
    @abstractmethod
    def getTopicLDA(self, index):
        '''
        Returns the most probable topic of a document with index index as per LDA
        '''
        
    @abstractmethod
    def getNamedEntities(self, index):
        '''
        Returns the named entities discovered in the document of index index
        '''
        














doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]
#
#from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer
#import string
#stop = set(stopwords.words('english'))
#exclude = set(string.punctuation)
#lemma = WordNetLemmatizer()
#def clean(doc):
#    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
#    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
#    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
#    return normalized
#
#doc_clean = [clean(doc).split() for doc in doc_complete]
#
## Importing Gensim
#import gensim
#from gensim import corpora
#
## Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
#
## Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
#dictionary = corpora.Dictionary(doc_clean)
#doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#
## Creating the object for LDA model using gensim library
#Lda = gensim.models.ldamodel.LdaModel
#
## Running and Trainign LDA model on the document term matrix.
#ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
#
#print(ldamodel.print_topics(num_topics=3, num_words=3))



#
#def get_tfidf(documents):  # ??gensim????tfidf
#    dictionary = gensim.corpora.Dictionary(documents)
#    n_items = len(dictionary)
#    corpus = [dictionary.doc2bow(text) for text in documents]
#    tfidf = gensim.models.TfidfModel(corpus)
#    corpus_tfidf = tfidf[corpus]
#
#    ds = []
#    for doc in corpus_tfidf:
#        [0] * n_items
#        for index, value in doc :
#            print(index, value)
#            d[index]  = value
#        ds.append(d)
#    return None
#
#get_tfidf(doc_clean)