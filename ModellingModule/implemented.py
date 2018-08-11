# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:49:25 2018

@author: c14238a


Notes:
    http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    for the complete NLTK tag list
    
    https://pythonprogramming.net/named-entity-recognition-stanford-ner-tagger/
    a quick start with the Stanford NER tagger
    
To Do:
    write unseen data prediction methods
"""

#import os
#path = "C://Work//Python//Articles//"
#path = "D://Книги//Мои//Python//Articles//Articles//"
#os.chdir(path)

from ModellingModule.abstract import *

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The number of topics for LDA and NMF
NUMBER_OF_TOPICS = 2
NUMBER_TOPIC_WORDS = 15

class InsiderNLP(NLPProcessor):
    '''
    This class is intended to process documents that come from insidermedia.com, 
    although it is implemented to be as general as possible.
    
    It implements the methods described in NLPProcessor, so please refer to it for
    more information.
    
    As a difference to the abstract class, the following is added. This class can
    load a model (the 'Database model') from the disk and use it to predict the
    data in self.
    
    A note, the codes for LDA and NMF modelling are candidates for refactoring - 
    it was just too much copying and pasting when writing them...
    '''
    
    def __init__(self, list_of_documents):
        '''
        The constructor stores the input list of documents in an object
        variable and instantiate the list of the cleaned documents.
        
        Parameters:
            list_of_documents: list; contains one or more strings of text.
        Returns:
            None; instantiates the object
        '''
        self.raw_documents = list_of_documents
        self.cleaned_documents = []
        self.lexicon = pd.read_excel('./ModellingModule/nrc.xlsx')
        
    def clean(self, split_each_document = False):
        '''
        This method cleans the raw data and store the cleaned data in the
        already initialized object variable.
        The cleaning process contains the following steps:
            1. Convert everything to lower case;
            2. Remove any stopwords;
            3. Remove punctuation;
            4. Tokenize.
            
        Parameters:
            split_each_document: boolean; True to output the cleaned documents
            into a list of lists, where each sublist contains the terms in a given document.
            
            The method also uses the raw data in self that is provided by __init__
            
        Returns:
            None; stores the cleaned data as an object variable
        '''
        # Initialize a temporary string and a list of the cleaned documents
        temp_doc = ''
        cleaned_documents = []
        
        # Initialize the stopwords list, the punctuation list and the WordNetLemmatizer
        stopwords_list = set(nltk.corpus.stopwords.words('english'))
        punctuation = set(string.punctuation)
        stem_finder = nltk.stem.wordnet.WordNetLemmatizer()
        
        # Loop over each document
        for document in self.raw_documents[:]:
            # Remove the stopwords, punctuation and lemmatize the words
            temp_doc = ' '.join([word for word in document.lower().split()
                                    if word not in stopwords_list])
            temp_doc = ''.join(char for char in temp_doc if char not in punctuation)
            temp_doc = ' '.join(stem_finder.lemmatize(word) for word in temp_doc.split())
        
            # Append to the cleaned document list
            cleaned_documents.append(temp_doc)
        
        # Split the documents into lists of strings
        if split_each_document is True:
            cleaned_documents = [[word for word in document.split()] 
                                for document in cleaned_documents]
            
        # Store the cleaned data as an object attribute
        self.cleaned_documents = cleaned_documents
        
    def analyzeFrequencies(self, n = 10, plot = False):
        '''
        This method takes the cleaned data and finds the most frequent words in
        it. If there is one document it will use standard frequencies; (not imple-
        mented --> ) if there are multiple - it will weight using Tf-Idf.
        
        And yes, of course I can use scikit-learn or gensim or nltk or whatever
        to do tf-idf, but it was more interesting to write it myself, however bad
        the code is.
        
        Parameters:
            n: int; the number of the most frequent words to display
            plot: boolean; True to produce a plot of the most frequent words
            
        Returns:
            pandas DataFrame with the most frequent words
        '''
        ### Calculate the term frequencies of each word within a document
        # Initalize some variables
        term_frequency_list = []
        temp_dict = {}
        
        # Loop over each document
        for document in self.cleaned_documents:
            # Re-initialize temp_dict, otherwise it'll not behave as expected
            temp_dict = {}
            # Loop over each word in the document
            for word in document.split():
            # Add it to the counter dictionary
                if word in temp_dict.keys():
                    temp_dict[word] += 1
                else:
                    temp_dict[word] = 1
            # Now calculate the term frequency of each word within the document
            for key, value in temp_dict.items():
                temp_dict[key] = value / len(document.split())
                
            term_frequency_list.append(temp_dict)    
            
            
        ### Calculate the inverse document frequencies
        # Create a idf dictionary - only one is needed for all documents altogether
        idf_dictionary = {}
        # Create a flattened list of all words and extract the unique words only
        unique_words_list = [word for document in self.cleaned_documents 
                                                     for word in document.split()]
        unique_words_list = set(unique_words_list)
        # Loop over each word in the unique-words-flattened-list
        for word in unique_words_list:
            # Loop over each document
            for document in self.cleaned_documents:
                # If the word is in the document add one to the corresponding idf dictionary value
                if word in document.split():
                    if word in idf_dictionary.keys():
                        idf_dictionary[word] += 1
                    else:
                        idf_dictionary[word] = 1
        
        
        ### Calculate the tf_idf frequencties
        # Initialize the final tf-idf list-of-dictionaries (for each document a 
        # separate dictionary is needed)
        tf_idf_list = []
        
        # Loop over each dictionary in the term frequencies dictionary list
        for dictionary in term_frequency_list:
            temp_dict = {}
            # Loop over each word in the dictionary
            for word in dictionary.keys():
                # Add a new key:value pair to the final dictionary by multiplying the
                # term frequency by the inverse document frequency
                if word in temp_dict.keys():
                    raise ValueError('Something unexpected occured in the tf * idf calculations loop')
                else:
                    temp_dict[word] = dictionary[word] * np.log(
                                                                len(self.cleaned_documents) / 
                                                                    idf_dictionary[word])
                    
            tf_idf_list.append(temp_dict)
        
        ### Store to object attributes
        self.term_frequency_list = term_frequency_list
        self.idf_dictionary = idf_dictionary
        self.tf_idf_list = tf_idf_list
     
        
        
        # Deprecated
        
#        ### Produce the most frequent words data frame
#        # Initialize an empty dictionary
#        term_counts = {}
#        # Unlist the list of lists into a single list
#        flattened_list = [word for sublist in self.cleaned_documents for word in sublist.split()]
#        # Loop over each term in the list
#        for word in flattened_list:
#            # If the key exists add one to the value
#            if word in term_counts:
#                term_counts[word] += 1
#            # Else create the key with value 0
#            else:
#                term_counts[word] = 1
#        # Create a data frame from the dict
#        frequency_df = pd.DataFrame({'terms':list(term_counts.keys()), 
#                                     'frequencies':list(term_counts.values())})
#        # Sort the data frame by frequency counts
#        frequency_df.sort_values(by = 'frequencies', axis = 0, ascending = True, inplace = True)
#        
#        ### If plot = True produce a plot
#        # Create the plot
#        plot = plt.barh(y = np.arange(n), 
#                       width = frequency_df.tail(n).frequencies)
#        plt.yticks(np.arange(n), frequency_df.tail(n).terms)
#        # Show the plot
#        plt.show(plot)
    
        
    def analyzeSentiment(self):
        '''
        This method takes the cleaned data and analyzes the polarity of the documents
        and the emotions present in them.
        
        Parameters:
            None; uses the cleaned data in self.
        Returns:
            dictionary; the first key:value pair is the polarity score, the second - 
            a table with the emotions values
        '''   
        # Initialize an empty polarity score list. It should be of the same length as the documents list.
        polarity_list = []
        emotions_tables = []
        
        # self.cleaned_documents is a list of lists, so loop over each sublist
        for document in self.cleaned_documents:
            # Convert the sublsit to pandas dataframe. Use a list comprehension!
            document_dataframe = pd.DataFrame({'document':[word for word in document.split()]})
            # Join the two datasets
            joined_dataframe = pd.merge(document_dataframe, self.lexicon,
                                        left_on = 'document', right_on = 'word')
            
            ### Polarity score
            
            # Filter only the positive/negative labels and make a frequency table of them
            positive_dataframe = joined_dataframe[joined_dataframe.sentiment == 'positive']
            negative_dataframe = joined_dataframe[joined_dataframe.sentiment == 'negative']
#            test_df = pd.concat([positive_dataframe, negative_dataframe], axis = 0)
#            print(test_df)
            # Get the polarity score
            if len(positive_dataframe) == 0 and len(negative_dataframe) == 0:
                print('No words matched!')
                polarity_list.append('nan')
            else:
                percent_positive = len(positive_dataframe) / (len(positive_dataframe) + len(negative_dataframe))
                polarity_list.append(percent_positive)
                
            ### Emotions
            joined_dataframe = joined_dataframe[joined_dataframe.sentiment != 'positive']
            joined_dataframe = joined_dataframe[joined_dataframe.sentiment != 'negative']
            
            emotions_table = joined_dataframe.sentiment.value_counts(normalize = True)
            emotions_tables.append(emotions_table)
            
        # Return
        self.polarity_scores = polarity_list
        self.emotions_tables = emotions_tables
#        print()
#        print(self.polarity_scores)
#        print()
#        print(self.emotions_tables)
#        print()
#        print(test_df.sentiment.value_counts(normalize = True))
        
        
        
    def doNER(self):
        '''
        Performs Named Entity Recognition on a given set of documents.
        
        Returns:
            stores to self a list of dictionaries; each list element corresponds
            to a document in self.raw_documents; the dictionary keys are the NER 
            labels, values are lists of discovered entites.
        '''
        # This is the list where the results for each document will be stored as dictionaries.
        named_entity_list = []
        
        ### Perform the Named entity recognition
        
        # Loop through each document in self.raw_documents
        for document in self.raw_documents:
            # We need the raw text for the chunking - self.cleaned_documents won't work
            # in this case. The below returns list of strings, where each string is
            # a sentence
            sentences = nltk.sent_tokenize(document)
            # Tokenize the sentences - returns list of lists, where each sublist contains
            # separate words as elements.
            tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            # Adds POS tags to the tokenized sentences
            tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            # Adds NER tags to the tagged sentences and returns a generator
            chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=False)
        
        
        ### Extract the named entities in each document and group them
            
            temp_dict = {'GPE':[],
                         'Organization':[],
                         'Person':[],
                         'Location':[],
                         'Other':[]} 
            # to avoid too much if statements, the structure of the dictionary is set here.
            
            # Loop over the chunked sentences
            for sentence in chunked_sentences:
                # Loop over the chunks in each sentence
                for chunk in sentence:
                    # Check if the chunk has a label
                    if hasattr(chunk, 'label'):
                        # Check what the label is and add it to its dictionary value
                        if chunk.label() == 'GPE':
                            temp_dict['GPE'].append(' '.join(c[0] for c in chunk))
                        elif chunk.label() == 'ORGANIZATION':
                            temp_dict['Organization'].append(' '.join(c[0] for c in chunk))
                        elif chunk.label() == 'PERSON':
                            temp_dict['Person'].append(' '.join(c[0] for c in chunk))
                        elif chunk.label() == 'LOCATION':
                            temp_dict['Location'].append(' '.join(c[0] for c in chunk))
                        else:
                            temp_dict['Other'].append(' '.join(c[0] for c in chunk))
                        
            # Add the dictionary to the list, but before that sort and remove duplicates
            for key, value in temp_dict.items():
                temp_dict[key] = sorted(list(set(value)))
                
            named_entity_list.append(temp_dict)
            
        # Store the list to self
        self.named_entities = named_entity_list
        
    def doLDA(self, save_model = False):
        '''
        This method performs Latent Dirichlet Allocation and assigns the most
        probable topic to each document.
        
        It joins first each document's list of words into a single string separated
        by white spaces, so that it can be directly used by CoundVectorizer.
        
        The Latend Dirichlet Allocation defaults to 10 topics, which is used here.
        It also uses sklearn's default values for LatentDirichletAllocation.
        '''
        # Convert the cleaned documents into list of strings (each string - a document)
        strings_list = [word for sublist in self.cleaned_documents for word in sublist.split()]
        
        # Initialize the CountVectorizer
        n_features = 1000 # why not...
        count_vectorizer = CountVectorizer(max_df=0.95, min_df=1,
                                           max_features = n_features)
        count_transform = count_vectorizer.fit_transform(strings_list)
        features = count_vectorizer.get_feature_names()
        
        # Initialize the LDA model and fit_transform the data
        lda = LatentDirichletAllocation(n_components = NUMBER_OF_TOPICS, learning_method = 'batch')
        lda_transform = lda.fit_transform(count_transform)
        
        # Store the relevant variables to self (storing the whole model is kind of ineficcient though...)
        lda_results = {'features': features, 'transform': lda_transform, 'model': lda}
        # Store the topics in a dictionary and then to self.nmf_results
        lda_topic_dict = {}
        for topic_index, topic in enumerate(lda.components_):
            lda_topic_dict[topic_index] = [features[i] for i in topic.argsort() \
                                          [:-NUMBER_TOPIC_WORDS - 1:-1]]
        lda_results['topics'] = lda_topic_dict
            
        self.lda_results = lda_results
        
        # If modelling the database, store a text file with the model details, so it can be used later.
        if save_model is True:
            model = (features, self.lda_results['model'])
            
            with open('.//ModellingModule//lda_whole_model', 'wb') as fp:
                joblib.dump(model, fp)
    
    def doNMF(self, save_model = False):
        '''
        This method performs Non-negative Matrix Factorization. Works in a similar
        fashion to the LDA above.
        '''
        ### Initial data processing
        # Convert the cleaned documents into list of strings
        strings_list = [word for sublist in self.cleaned_documents for word in sublist.split()]
        
        # Initialize the Tf-Idf Vectorizer
        n_features = 1000
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features)
        tfidf_transform = tfidf_vectorizer.fit_transform(strings_list)
        features = tfidf_vectorizer.get_feature_names()
        
        ### Modelling
        # Initialize the NMF model and fit_transform
        nmf = NMF(n_components = NUMBER_OF_TOPICS, beta_loss = 'kullback-leibler',
                  solver = 'mu')#, random_state=1, alpha=.1, 
#                  l1_ratio=.5, init='nndsvd')
        #Above the example parameters have been restored to the defaults and KL divergence used.
        
        nmf_transform = nmf.fit_transform(tfidf_transform)
        
        ### Results processing
        # Store the relevant variables to self
        nmf_results = {'features': features, 'transform': nmf_transform, 'model': nmf}
        # Store the topics in a dictionary and then to self.nmf_results
        nmf_topic_dict = {}
        for topic_index, topic in enumerate(nmf.components_):
            nmf_topic_dict[topic_index] = [features[i] for i in topic.argsort() \
                                          [:-NUMBER_TOPIC_WORDS - 1:-1]]
        nmf_results['topics'] = nmf_topic_dict
                
        self.nmf_results = nmf_results
        
        # Store a text file with the featueres and the model details as a tuple
        if save_model is True:
            model = (features, self.nmf_results['model'])
            with open('.//ModellingModule//nmf_whole_model', 'wb') as fp:
                joblib.dump(model, fp)
        
    def predictTopicLDA(self):
        '''
        So far the concept is that this method loads the model saved to disk (
        currently no functionality for specifying the path to the model, i.e. the
        user won't be able to change the model to be used) and predicts the topic
        of the data stored in self.
        I.e., first, this is NOT a function, where you specify the data and the
        location of the model and the function returns predictions;
        second, it doesn't overwrite the raw_documents in self. This is not the point.
        
        In one word, the data is fixed. However, it can be modelled either by
        creating a new model using the data (where the data becomes training data),
        or the data is used as test data by the Database model.
        
        And, again, the Database model is considered to be only one - this is the
        model that has been trained on the whole database. It can be used to predict
        newly downloaded articles.
        
        Anyway, enough with the philosophy; just write something that 1. takes
        a model from a file; 2. predicts the topics of the documents in self;
        3. returns the topics as a list of words.
        -----------------------------------------------------------------------
        Buuut... all this is fine, except that when loading a model it overwrites
        the existing model in self. This might not be a desired behavior, as the
        user may want to have both models in self, so that he can compare (e.g.).
        Also, this poses some issues when using the class in the kivy application, as
        the application is going to use results of both the database model and 
        the current model simultaneously in order to fill in the Labels in the GUI.
        
        An obvious alternative is, of course, to instantiate two NLP processor
        objects that are going to be used simultaneously with the same data. This 
        isn't any efficient as well, but it might be the better option. The problem
        with the first one is that the Getters won't know (refactor this if you dare...)
        which self.{model}_results to use - refactoring them in this case is, of
        course, possible, but it might be quicker just to use two separate objects
        no matter how inefficient is this. 
        An efficient solution will probably require more serious refactoring.
        '''
        # load parameters from file
        with open ('.//ModellingModule//lda_whole_model', 'rb') as fd:
            features, lda = joblib.load(fd)

        # Convert the cleaned documents into list of strings (each string - a document)
        strings_list = [word for sublist in self.cleaned_documents for word in sublist.split()]
        
        # Initialize the CountVectorizer
        count_vectorizer = CountVectorizer(vocabulary = features)
        count_transform = count_vectorizer.fit_transform(strings_list)
        
        # Predict
        lda_transform = lda.transform(count_transform)
        
        # Store the relevant variables to self (storing the whole model is kind of ineficcient though...)
        lda_results = {'features': features, 'transform': lda_transform, 'model': lda}
        # Store the topics in a dictionary and then to self.nmf_results
        lda_topic_dict = {}
        for topic_index, topic in enumerate(lda.components_):
            lda_topic_dict[topic_index] = [features[i] for i in topic.argsort() \
                                          [:-NUMBER_TOPIC_WORDS - 1:-1]]
        lda_results['topics'] = lda_topic_dict
            
        self.lda_results = lda_results
        
    def predictTopicNMF(self):
        '''
        The same as above, but uses NMF.
        '''
        # load parameters from file
        with open ('.//ModellingModule//nmf_whole_model', 'rb') as fd:
            features, nmf = joblib.load(fd)

        # Convert the cleaned documents into list of strings (each string - a document)
        strings_list = [word for sublist in self.cleaned_documents for word in sublist.split()]
        
        # Initialize the CountVectorizer
        tfidf_vectorizer = TfidfVectorizer(vocabulary = features)
        tfidf_transform = tfidf_vectorizer.fit_transform(strings_list)
        
        # Predict
        nmf_transform = nmf.transform(tfidf_transform)
        
        # Store the relevant variables to self (storing the whole model is kind of ineficcient though...)
        nmf_results = {'features': features, 'transform': nmf_transform, 'model': nmf}
        # Store the topics in a dictionary and then to self.nmf_results
        nmf_topic_dict = {}
        for topic_index, topic in enumerate(nmf.components_):
            nmf_topic_dict[topic_index] = [features[i] for i in topic.argsort() \
                                          [:-NUMBER_TOPIC_WORDS - 1:-1]]
        nmf_results['topics'] = nmf_topic_dict
            
        self.nmf_results = nmf_results
        
    def doAll(self, nlp_fit = True):
        '''
        This just runs all modelling methods above.
        
        Parameters:
            nlp_fit: boolean; True to fit the data in self using LDA/NMF. If True, 
            it is going to overwrite the saved models in the folder, so use with caution.
        
        Returns:
            None; saves to self the results of the modelling - use the Getters.
        '''
        self.clean()
        self.analyzeFrequencies()
        self.analyzeSentiment()
        self.doNER()
        if nlp_fit is True:
            self.doLDA()
            self.doNMF()
        else:
            self.predictTopicLDA()
            self.predictTopicNMF()
        
    def setRawData(self, list_of_documents):
        '''
        Sets new raw documents list.
        
        Parameters:
            list_of_documents: list; each element is a string that is considered
            a 'document'
            
        Returns:
            None
        '''
        self.raw_documents = list_of_documents        
        
    def getFrequencies(self, index, n, plot = False):
        '''
        The Getter for the word frequencies.
        
        Parameters:
            index: integer; a valid document index.
            n: integer, the number of most frequent words to be returned.
            
        Returns:
            dict; key is a word; value is the tf-idf weighted frequency
        '''
        # Extract the n largest values from the appropriate dictionary
        temp_dict = {key: self.tf_idf_list[index][key] for key in sorted(self.tf_idf_list[index], 
                     key = self.tf_idf_list[index].get, reverse = True)[:n]}
        # Sort them
        temp_dict = {key: temp_dict[key] for key in sorted(temp_dict, 
                     key = temp_dict.get, reverse = True)}
    
        # Make a plot, because - why not?
        if plot == True:
            plt.barh(range(len(temp_dict)), list(temp_dict.values()), align = 'center')
            plt.yticks(range(len(temp_dict)), list(temp_dict.keys()))
            plt.show()
        # Return
        return temp_dict
        
    def getNamedEntities(self, index, entity_name = ''):
        '''
        The Getter for the named entities.
        
        Parameters:
            index: integer; a valid document index.
        '''
        # Check if entity_name exists and is valid
        if entity_name == '': # return all entities for the selected index
            return self.named_entities[index]
        elif entity_name in self.named_entities[index].keys(): # return the desired entity list
            return self.named_entities[index][entity_name]
        else:
            raise ValueError('The entity name supplied is not valid.')
        
    def getPolarityScore(self, index):
        '''
        The Getter for the polarity score.
        
        Parameters:
            index: integer; a valid document index.
        
        Returns:
            float; the polarity score for the document of interest.
        '''
        return self.polarity_scores[index]
        
    def getSentiments(self, index):
        '''
        The Getter for the emotions.
        
        Parameters:
            index: integer; a valid document index.
            
        Returns:
            pandas.Series; a frequency table with the emotions counts
        '''
        return self.emotions_tables[index]
    
    def getTopicLDA(self, index):
        '''
        The Getter for the topic keywords.
        
        Parameters:
            index: integer; a valid document index.
            
        Returns:
            list; a list with the topic words
        '''
        # Extract the most probable topic
        most_probable_topic = self.lda_results['transform'][index].argmax()
        # Get the topic keywords (say, 10 most important)
        words_list = [
                      self.lda_results['features'][i] for i in self.lda_results \
                          ['model'].components_[most_probable_topic].argsort() \
                          [:-NUMBER_TOPIC_WORDS - 1:-1]
                     ]
        # Return something
        return words_list
    
    def getTopicNMF(self, index):
        '''
        The Getter for the topic keywords.
        
        Parameters:
            index: integer; a valid document index.
            
        Returns:
            list; a list with the topic words
        '''
        # Extract the most probable topic
        most_probable_topic = self.nmf_results['transform'][index].argmax()
        # Get the topic keywords (say, 10 most important)
        words_list = [
                      self.nmf_results['features'][i] for i in self.nmf_results \
                          ['model'].components_[most_probable_topic].argsort() \
                          [:-NUMBER_TOPIC_WORDS - 1:-1]
                     ]
        # Return something
        return words_list    
    
    def getAllTopics(self, model_type):
        '''
        Returns all topics and their keywords for a given model.
        
        Parameters:
            model_type: string; either "nmf" or "lda".
            
        Returns:
            Dictionary; key: topic number; value: list with NUMBER_TOPIC_WORDS top words
            Prints to the console.
        '''
        if model_type == 'nmf':
            print('========== NMF topics: ==========')
            for topic_index, topic in enumerate(self.nmf_results['model'].components_):
                message = "Topic #%d: " % topic_index
                message += " ".join([self.nmf_results['features'][i]
                                     for i in topic.argsort()[:-15 - 1:-1]])
                print(message)
            print()
            # Return the dictionary
            return self.nmf_results['topics']
            
        elif model_type == 'lda':
            print('========== LDA topics: ==========')
            for topic_index, topic in enumerate(self.lda_results['model'].components_):
                message = "Topic #%d: " % topic_index
                message += " ".join([self.lda_results['features'][i]
                                     for i in topic.argsort()[:-15 - 1:-1]])
                print(message)
            print()
            # Return the dictionary
            return self.lda_results['topics']
        
        else:
            print('Please choose either "nmf" or "lda".')
            return
        
        
##### Tests
#titles = ["This is some title saying that Bob likes Jane so much and he can't stop thinking about her.",
#          "This is another title that is not quite sure - the title is not sure I mean - what it is about.",
#          "Hey, you aren't the smartest person in the world, are you?",
#          "Actually, I'm Einstein. Really - I downloaded his mind from the database and installed it in my own mind",
#          "Are you watching Westworld? This is a nice show, but I'm not quite sure if it is going to be any interesting when they make the tenth or so season.",
#          "Oh, well, I actually used some Tesla technologies  - from Nicola Tesla, not that newcomer Musk guy - and indeed downloaded Einstein's mind into my own, so yeah - I'm Einstein. Do you want a brief Theory of Relativity introduction?",
#          "You are an idiot.",
#          "Yes, sir, I am.",
#          "Anakin said: I hate you!",
#          "I will bring peace, justice and stability to my new empire!",
#          "Your new empire? Anakin, my allegeance is to the republic, to democracy!",
#          'test test test',
#          'test test',
#          'test']
#a = InsiderNLP(titles)
#a.doAll()
#a.getAllTopics('lda')
#a.getFrequencies(n=15, index=1)
#a.getNamedEntities(1)
#a.getPolarityScore(1)
#a.getSentiments(1)
#a.clean()
##print(a.analyzeFrequencies())
##print(a.getFrequencies(1, 10))
##a.analyzeSentiment()
#a.doLDA(True)
#a.getTopicLDA(1)
#a._getAllTopics('lda')
#a.doNMF()
#a.getTopicNMF(1)
#a._getAllTopics('nmf')

## some wikipedia texts
#texts = ["Russia (Russian: Росси́я, tr. Rossiya, IPA: [rɐˈsʲijə]), officially the Russian Federation[12] (Russian: Росси́йская Федера́ция, tr. Rossiyskaya Federatsiya, IPA: [rɐˈsʲijskəjə fʲɪdʲɪˈratsɨjə]), is a country in Eurasia.[13] At 17,125,200 square kilometres (6,612,100 sq mi),[14] Russia is the largest country in the world by area, covering more than one-eighth of the Earth's inhabited land area,[15][16][17] and the ninth most populous, with over 144 million people as of December 2017, excluding Crimea.[8] About 77% of the population live in the western, European part of the country. Russia's capital Moscow is one of the largest cities in the world; other major cities include Saint Petersburg, Novosibirsk, Yekaterinburg and Nizhny Novgorod. Extending across the entirety of Northern Asia and much of Eastern Europe, Russia spans eleven time zones and incorporates a wide range of environments and landforms. From northwest to southeast, Russia shares land borders with Norway, Finland, Estonia, Latvia, Lithuania and Poland (both with Kaliningrad Oblast), Belarus, Ukraine, Georgia, Azerbaijan, Kazakhstan, China, Mongolia and North Korea. It shares maritime borders with Japan by the Sea of Okhotsk and the U.S. state of Alaska across the Bering Strait. The East Slavs emerged as a recognizable group in Europe between the 3rd and 8th centuries AD.[18] Founded and ruled by a Varangian warrior elite and their descendants, the medieval state of Rus arose in the 9th century. In 988 it adopted Orthodox Christianity from the Byzantine Empire,[19] beginning the synthesis of Byzantine and Slavic cultures that defined Russian culture for the next millennium.[19] Rus' ultimately disintegrated into a number of smaller states; most of the Rus' lands were overrun by the Mongol invasion and became tributaries of the nomadic Golden Horde in the 13th century.[20] The Grand Duchy of Moscow gradually reunified the surrounding Russian principalities, achieved independence from the Golden Horde. By the 18th century, the nation had greatly expanded through conquest, annexation, and exploration to become the Russian Empire, which was the third largest empire in history, stretching from Poland on the west to Alaska on the east.[21][22] Following the Russian Revolution, the Russian Soviet Federative Socialist Republic became the largest and leading constituent of the Union of Soviet Socialist Republics, the world's first constitutionally socialist state.[23] The Soviet Union played a decisive role in the Allied victory in World War II,[24][25] and emerged as a recognized superpower and rival to the United States during the Cold War. The Soviet era saw some of the most significant technological achievements of the 20th century, including the world's first human-made satellite and the launching of the first humans in space. By the end of 1990, the Soviet Union had the world's second largest economy, largest standing military in the world and the largest stockpile of weapons of mass destruction.[26][27][28] Following the dissolution of the Soviet Union in 1991, twelve independent republics emerged from the USSR: Russia, Ukraine, Belarus, Kazakhstan, Uzbekistan, Armenia, Azerbaijan, Georgia, Kyrgyzstan, Moldova, Tajikistan, Turkmenistan and the Baltic states regained independence: Estonia, Latvia, Lithuania; the Russian SFSR reconstituted itself as the Russian Federation and is recognized as the continuing legal personality and a successor of the Soviet Union.[29] It is governed as a federal semi-presidential republic. The Russian economy ranks as the twelfth largest by nominal GDP and sixth largest by purchasing power parity in 2015.[30] Russia's extensive mineral and energy resources are the largest such reserves in the world,[31] making it one of the leading producers of oil and natural gas globally.[32][33] The country is one of the five recognized nuclear weapons states and possesses the largest stockpile of weapons of mass destruction.[34] Russia is a great power as well as a regional power and has been characterised as a potential superpower. It is a permanent member of the United Nations Security Council and an active global partner of ASEAN,[35][36][37] as well as a member of the G20, the Shanghai Cooperation Organisation (SCO), the Council of Europe, the Asia-Pacific Economic Cooperation (APEC), the Organization for Security and Co-operation in Europe (OSCE), and the World Trade Organization (WTO), as well as being the leading member of the Commonwealth of Independent States (CIS), the Collective Security Treaty Organization (CSTO) and one of the five members of the Eurasian Economic Union (EEU), along with Armenia, Belarus, Kazakhstan and Kyrgyzstan. ",
#         "The United States of America (USA), commonly known as the United States (U.S.) or America, is a federal republic composed of 50 states, a federal district, five major self-governing territories, and various possessions.[fn 6] At 3.8 million square miles (9.8 million km2) and with over 325 million people, the United States is the world's third- or fourth-largest country by total area[fn 7] and the third-most populous country. The capital is Washington, D.C., and the largest city by population is New York City. Forty-eight states and the capital's federal district are contiguous in North America between Canada and Mexico. The State of Alaska is in the northwest corner of North America, bordered by Canada to the east and across the Bering Strait from Russia to the west. The State of Hawaii is an archipelago in the mid-Pacific Ocean. The U.S. territories are scattered about the Pacific Ocean and the Caribbean Sea, stretching across nine official time zones. The extremely diverse geography, climate, and wildlife of the United States make it one of the world's 17 megadiverse countries.[19] Paleo-Indians migrated from Asia to the North American mainland at least 15,000 years ago.[20] European colonization began in the 16th century. The United States emerged from the thirteen British colonies established along the East Coast. Numerous disputes between Great Britain and the colonies following the French and Indian War led to the American Revolution, which began in 1775, and the subsequent Declaration of Independence in 1776. The war ended in 1783 with the United States becoming the first country to gain independence from a European power.[21] The current constitution was adopted in 1788, with the first ten amendments, collectively named the Bill of Rights, being ratified in 1791 to guarantee many fundamental civil liberties. The United States embarked on a vigorous expansion across North America throughout the 19th century, acquiring new territories,[22] displacing Native American tribes, and gradually admitting new states until it spanned the continent by 1848.[22] During the second half of the 19th century, the Civil War led to the abolition of slavery.[23][24] By the end of the century, the United States had extended into the Pacific Ocean,[25] and its economy, driven in large part by the Industrial Revolution, began to soar.[26] The Spanish–American War and World War I confirmed the country's status as a global military power. The United States emerged from World War II as a global superpower, the first country to develop nuclear weapons, the only country to use them in warfare, and a permanent member of the United Nations Security Council. During the Cold War, the United States and the Soviet Union competed in the Space Race, culminating with the 1969 moon landing. The end of the Cold War and the collapse of the Soviet Union in 1991 left the United States as the world's sole superpower.[27] The United States is the world's oldest surviving federation. It is a representative democracy, in which majority rule is tempered by minority rights protected by law.[28] The United States is a founding member of the United Nations, World Bank, International Monetary Fund, Organization of American States (OAS), and other international organizations. The United States is a highly developed country, with the world's largest economy by nominal GDP and second-largest economy by PPP, accounting for approximately a quarter of global GDP.[29] The U.S. economy is largely post-industrial, characterized by the dominance of services and knowledge-based activities, although the manufacturing sector remains the second-largest in the world.[30] The United States is the world's largest importer and the second largest exporter of goods.[31][32] Though its population is only 4.3% of the world total,[33] the U.S. holds 33.4% of the total wealth in the world, the largest share of global wealth concentrated in a single country.[34] The United States ranks among the highest nations in several measures of socioeconomic performance, including average wage,[35] human development, per capita GDP, and productivity per person.[36] The U.S. is the foremost military power in the world, making up a third of global military spending,[37] and is a leading political, cultural, and scientific force internationally.[38]",
#         "China, officially the People's Republic of China (PRC), is a unitary one-party sovereign state in East Asia and the world's most populous country, with a population of around 1.404 billion.[13] Covering approximately 9,600,000 square kilometers (3,700,000 sq mi), it is the third- or fourth-largest country by total area,[k][19] depending on the source consulted. Governed by the Communist Party of China, it exercises jurisdiction over 22 provinces, five autonomous regions, four direct-controlled municipalities (Beijing, Tianjin, Shanghai, and Chongqing), and the special administrative regions of Hong Kong and Macau. China emerged as one of the world's earliest civilizations, in the fertile basin of the Yellow River in the North China Plain. For millennia, China's political system was based on hereditary monarchies, or dynasties, beginning with the semi-legendary Xia dynasty in 21st century BCE.[20] Since then, China has expanded, fractured, and re-unified numerous times. In the 3rd century BCE, the Qin unified core China and established the first Chinese dynasty. The succeeding Han dynasty, which ruled from 206 BC until 220 AD, saw some of the most advanced technology at that time, including papermaking and the compass,[21] along with agricultural and medical improvements. The invention of gunpowder and printing in the Tang dynasty (618 - 907) completed the Four Great Inventions. Tang culture spread widely in Asia, as the new maritime Silk Route brought traders to as far as Mesopotamia and Somalia.[22] Dynastic rule ended in 1912 with the Xinhai Revolution, as a republic replaced the Qing dynasty. The Chinese Civil War led to the break up of the country in 1949, with the victorious Communist Party of China founding the People’s Republic of China on the mainland while the losing Kuomintang retreated to Taiwan, a dispute which is still unresolved. Since the introduction of economic reforms in 1978, China's economy has been one of the world's fastest-growing with annual growth rates consistently above 6 percent.[23] As of 2016, it is the world's second-largest economy by nominal GDP and largest by purchasing power parity (PPP).[24] China is also the world's largest exporter and second-largest importer of goods.[25] China is a recognized nuclear weapons state and has the world's largest standing army and second-largest defense budget.[26][27][28] The PRC is a member of the United Nations, as it replaced the ROC as a permanent member of the UN Security Council in 1971. China is also a member of numerous formal and informal multilateral organizations, including the ASEAN Plus mechanism, WTO, APEC, BRICS, the Shanghai Cooperation Organization (SCO), the BCIM, and the G20. China is a great power and a major regional power within Asia, and has been characterized as a potential superpower.[29][30]",     
#         "India (IAST: Bhārat), also called the Republic of India (IAST: Bhārat Gaṇarājya),[19][e] is a country in South Asia. It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world. It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast. It shares land borders with Pakistan to the west;[f] China, Nepal, and Bhutan to the northeast; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives. India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia. The Indian subcontinent was home to the urban Indus Valley Civilisation of the 3rd millennium BCE. In the following millennium, the oldest scriptures associated with Hinduism began to be composed. Social stratification, based on caste, emerged in the first millennium BCE, and Buddhism and Jainism arose. Early political consolidations took place under the Maurya and Gupta empires; the later peninsular Middle Kingdoms influenced cultures as far as southeast Asia. In the medieval era, Judaism, Zoroastrianism, Christianity, and Islam arrived, and Sikhism emerged, all adding to the region's diverse culture. Much of the north fell to the Delhi sultanate; the south was united under the Vijayanagara Empire. The economy expanded in the 17th century in the Mughal Empire. In the mid-18th century, the subcontinent came under British East India Company rule, and in the mid-19th under British crown rule. A nationalist movement emerged in the late 19th century, which later, under Mahatma Gandhi, was noted for nonviolent resistance and led to India's independence in 1947. In 2017, the Indian economy was the world's sixth largest by nominal GDP[20] and third largest by purchasing power parity.[16] Following market-based economic reforms in 1991, India became one of the fastest-growing major economies and is considered a newly industrialised country. However, it continues to face the challenges of poverty, corruption, malnutrition, and inadequate public healthcare. A nuclear weapons state and regional power, it has the second largest standing army in the world and ranks fifth in military expenditure among nations. India is a federal republic governed under a parliamentary system and consists of 29 states and 7 union territories. It is a pluralistic, multilingual and multi-ethnic society and is also home to a diversity of wildlife in a variety of protected habitats.",
#         "Germany (German: Deutschland [ˈdɔʏtʃlant]), officially the Federal Republic of Germany (German: Bundesrepublik Deutschland, About this sound listen (help·info)),[e][9] is a sovereign state in central-western Europe. It includes 16 constituent states, covers an area of 357,386 square kilometres (137,988 sq mi),[4] and has a largely temperate seasonal climate. With nearly 83 million inhabitants, Germany is the most populous member state of the European Union. Germany's capital and largest metropolis is Berlin, while its largest conurbation is the Ruhr, with its main centres of Dortmund and Essen. The country's other major cities are Hamburg, Munich, Cologne, Frankfurt, Stuttgart, Düsseldorf, Leipzig, Bremen, Dresden, Hannover, and Nuremberg. Various Germanic tribes have inhabited the northern parts of modern Germany since classical antiquity. A region named Germania was documented before 100 AD. During the Migration Period, the Germanic tribes expanded southward. Beginning in the 10th century, German territories formed a central part of the Holy Roman Empire.[10] During the 16th century, northern German regions became the centre of the Protestant Reformation. After the collapse of the Holy Roman Empire, the German Confederation was formed in 1815. The German revolutions of 1848–49 resulted in the Frankfurt Parliament establishing major democratic rights. In 1871, Germany became a nation state when most of the German states unified into the Prussian-dominated German Empire. After World War I and the revolution of 1918–19, the Empire was replaced by the parliamentary Weimar Republic. The Nazi seizure of power in 1933 led to the establishment of a dictatorship, World War II and the Holocaust. After the end of World War II in Europe and a period of Allied occupation, two German states were founded: West Germany, formed from the American, British, and French occupation zones, and East Germany, formed from the Soviet occupation zone. Following the Revolutions of 1989 that ended communist rule in Central and Eastern Europe, the country was reunified on 3 October 1990.[11] In the 21st century, Germany is a great power with a strong economy; it has the world's fourth-largest economy by nominal GDP, and the fifth-largest by PPP. As a global leader in several industrial and technological sectors, it is both the world's third-largest exporter and importer of goods. A developed country with a very high standard of living, it upholds a social security and universal health care system, environmental protection, and a tuition-free university education. The Federal Republic of Germany was a founding member of the European Economic Community in 1957 and the European Union in 1993. It is part of the Schengen Area and became a co-founder of the Eurozone in 1999. Germany is a member of the United Nations, NATO, the G7, the G20, and the OECD. Known for its rich cultural history, Germany has been continuously the home of influential and successful artists, philosophers, musicians, sportspeople, entrepreneurs, scientists, engineers, and inventors.",
#         "South Africa, officially the Republic of South Africa (RSA), is the southernmost country in Africa. It is bounded to the south by 2,798 kilometres (1,739 mi) of coastline of Southern Africa stretching along the South Atlantic and Indian Oceans;[9][10][11] to the north by the neighbouring countries of Namibia, Botswana, and Zimbabwe; and to the east and northeast by Mozambique and Swaziland (Eswatini); and it surrounds the kingdom of Lesotho.[12] South Africa is the largest country in Southern Africa[13] and the 25th-largest country in the world by land area and, with close to 56 million people, is the world's 24th-most populous nation. It is the southernmost country on the mainland of the Old World or the Eastern Hemisphere. About 80 percent of South Africans are of Sub-Saharan African ancestry,[5] divided among a variety of ethnic groups speaking different African languages, nine of which have official status.[11] The remaining population consists of Africa's largest communities of European (white), Asian (Indian), and multiracial (Coloured) ancestry. South Africa is a multiethnic society encompassing a wide variety of cultures, languages, and religions. Its pluralistic makeup is reflected in the constitution's recognition of 11 official languages, which is among the highest number of any country in the world.[11] Two of these languages are of European origin: Afrikaans developed from Dutch and serves as the first language of most white and coloured South Africans; English reflects the legacy of British colonialism, and is commonly used in public and commercial life, though it is fourth-ranked as a spoken first language.[11] The country is one of the few in Africa never to have had a coup d'état, and regular elections have been held for almost a century. However, the vast majority of black South Africans were not enfranchised until 1994. During the 20th century, the black majority sought to recover its rights from the dominant white minority, with this struggle playing a large role in the country's recent history and politics. The National Party imposed apartheid in 1948, institutionalising previous racial segregation. After a long and sometimes violent struggle by the African National Congress and other anti-apartheid activists both inside and outside the country, the repeal of discriminatory laws began in 1990. Since 1994, all ethnic and linguistic groups have held political representation in the country's democracy, which comprises a parliamentary republic and nine provinces. South Africa is often referred to as the rainbow nation to describe the country's multicultural diversity, especially in the wake of apartheid.[14] The World Bank classifies South Africa as an upper-middle-income economy, and a newly industrialised country.[15][16] Its economy is the second-largest in Africa, and the 34th-largest in the world.[6] In terms of purchasing power parity, South Africa has the seventh-highest per capita income in Africa. However, poverty and inequality remain widespread, with about a quarter of the population unemployed and living on less than US$1.25 a day.[17][18] Nevertheless, South Africa has been identified as a middle power in international affairs, and maintains significant regional influence.[19][20]",
#         "Iran (Persian: ایران‎ Irān [ʔiːˈɾɒːn] (About this sound listen)), also known as Persia[10][11][12] (/ˈpɜːrʒə/),[13] officially the Islamic Republic of Iran (Persian: جمهوری اسلامی ایران‎ Jomhuri-ye Eslāmi-ye Irān (About this sound listen)),[14] is a sovereign state in Western Asia.[15][16] With over 81 million inhabitants,[6] Iran is the world's 18th-most-populous country.[17] Comprising a land area of 1,648,195 km2 (636,372 sq mi), it is the second-largest country in the Middle East and the 17th-largest in the world. Iran is bordered to the northwest by Armenia and the Republic of Azerbaijan,[a] to the north by the Caspian Sea, to the northeast by Turkmenistan, to the east by Afghanistan and Pakistan, to the south by the Persian Gulf and the Gulf of Oman, and to the west by Turkey and Iraq. The country's central location in Eurasia and Western Asia, and its proximity to the Strait of Hormuz, give it geostrategic importance.[18] Tehran is the country's capital and largest city, as well as its leading economic and cultural center. Iran is home to one of the world's oldest civilizations,[19][20] beginning with the formation of the Elamite kingdoms in the fourth millennium BCE. It was first unified by the Iranian Medes in the seventh century BCE,[21] reaching its greatest territorial size in the sixth century BCE, when Cyrus the Great founded the Achaemenid Empire, which stretched from Eastern Europe to the Indus Valley, becoming one of the largest empires in history.[22] The Iranian realm fell to Alexander the Great in the fourth century BCE and was divided into several Hellenistic states. An Iranian rebellion culminated in the establishment of the Parthian Empire, which was succeeded in the third century CE by the Sasanian Empire, a leading world power for the next four centuries.[23][24] Arab Muslims conquered the empire in the seventh century CE, displacing the indigenous faiths of Zoroastrianism and Manichaeism with Islam. Iran made major contributions to the Islamic Golden Age that followed, producing many influential figures in art and science. After two centuries, a period of various native Muslim dynasties began, which were later conquered by the Turks and the Mongols. The rise of the Safavids in the 15th century led to the reestablishment of a unified Iranian state and national identity,[4] with the country's conversion to Shia Islam marking a turning point in Iranian and Muslim history.[5][25] Under Nader Shah, Iran was one of the most powerful states in the 18th century,[26] though by the 19th century, a series of conflicts with the Russian Empire led to significant territorial losses.[27][28] Popular unrest led to the establishment of a constitutional monarchy and the country's first legislature. A 1953 coup instigated by the United Kingdom and the United States resulted in greater autocracy and growing anti-Western resentment.[29] Subsequent unrest against foreign influence and political repression led to the 1979 Revolution and the establishment of an Islamic republic,[30] a political system that includes elements of a parliamentary democracy vetted and supervised by a theocracy governed by an autocratic Supreme Leader.[31] During the 1980s, the country was engaged in a war with Iraq, which lasted for almost nine years and resulted in a high number of casualties and economic losses for both sides. According to international reports, Iran's human rights record is exceptionally poor. The regime in Iran is undemocratic,[32] and has frequently persecuted and arrested critics of the government and its Supreme Leader. Women's rights in Iran are described as seriously inadequate,[33] and children's rights have been severely violated, with more child offenders being executed in Iran than in any other country in the world.[34][35] Since the 2000s, Iran's controversial nuclear program has raised concerns, which is part of the basis of the international sanctions against the country. The Joint Comprehensive Plan of Action, an agreement reached between Iran and the P5+1, was created on 14 July 2015, aimed to loosen the nuclear sanctions in exchange for Iran's restriction in producing enriched uranium. Iran is a founding member of the UN, ECO, NAM, OIC, and OPEC. It is a major regional and middle power,[36][37] and its large reserves of fossil fuels – which include the world's largest natural gas supply and the fourth-largest proven oil reserves[38][39] – exert considerable influence in international energy security and the world economy. The country's rich cultural legacy is reflected in part by its 22 UNESCO World Heritage Sites, the third-largest number in Asia and eleventh-largest in the world.[40] Iran is a multicultural country comprising numerous ethnic and linguistic groups, the largest being Persians (61%), Azeris (16%), Kurds (10%), and Lurs (6%).[2]",
#         "Italy (Italian: Italia [iˈtaːlja] (About this sound listen)), officially the Italian Republic (Italian: Repubblica Italiana [reˈpubblika itaˈljaːna]),[10][11][12][13] is a sovereign state in Europe. Located in the heart of the Mediterranean Sea, Italy shares open land borders with France, Switzerland, Austria, Slovenia, San Marino and Vatican City. Italy covers an area of 301,340 km2 (116,350 sq mi) and has a largely temperate seasonal and Mediterranean climate. With around 61 million inhabitants, it is the fourth-most populous EU member state and the most populous in southern Europe. Since classical times, ancient Phoenicians, Carthaginians and Greeks established settlements in the south of Italy, with Etruscans and Celts inhabiting the centre and the north of Italy respectively, and various ancient Italian tribes and Italic peoples dispersed throughout the Italian Peninsula and insular Italy. The Italic tribe known as the Latins formed the Roman Kingdom in the 8th century BC, which eventually became a republic that conquered and assimilated its neighbours. Ultimately, the Roman Empire emerged in the 1st century BC as the dominant power in the Mediterranean Basin and became the leading cultural, political and religious centre of Western civilisation. The legacy of the Roman Empire is widespread and can be observed in the global distribution of civilian law, republican governments, Christianity and the Latin script. During the Early Middle Ages, Italy suffered sociopolitical collapse amid calamitous barbarian invasions, but by the 11th century numerous rival city-states and maritime republics, mainly in the northern and central regions of Italy, rose to great prosperity through shipping, commerce and banking, laying the groundwork for modern capitalism.[14] These mostly independent statelets, acting as Europe's main spice trade hubs with Asia and the Near East, often enjoyed a greater degree of democracy than the larger feudal monarchies that were consolidating throughout Europe; however, part of central Italy was under the control of the theocratic Papal States, while Southern Italy remained largely feudal until the 19th century, partially as a result of a succession of Byzantine, Arab, Norman, Angevin and Spanish conquests of the region.[15] The Renaissance began in Italy and spread to the rest of Europe, bringing a renewed interest in humanism, science, exploration and art. Italian culture flourished at this time, producing famous scholars, artists and polymaths, such as Michelangelo, Leonardo da Vinci, Raphael, Galileo and Machiavelli. Since the Middle Ages, Italian explorers such as Marco Polo, Christopher Columbus, Amerigo Vespucci, John Cabot and Giovanni da Verrazzano discovered new routes to the Far East and the New World, helping to usher in the European Age of Discovery. Nevertheless, Italy's commercial and political power significantly waned with the opening of trade routes which bypassed the Mediterranean.[15][16][17] Furthermore, the Italian city-states constantly engaged one another in bloody warfare, culminating in the Italian Wars of the 15th and 16th centuries that left them exhausted, with none emerging as a dominant power. They soon fell victim to conquest by European powers such as France, Spain and Austria. By the mid-19th century, a rising movement in support of Italian nationalism and independence from foreign control led to a period of revolutionary political upheaval. After centuries of foreign domination and political division, Italy was almost entirely unified in 1871, creating a great power.[18] From the late 19th century to the early 20th century, the new Kingdom of Italy rapidly industrialised, although mainly in the north, and acquired a colonial empire,[19] while the south remained largely impoverished and excluded from industrialisation, fuelling a large and influential diaspora.[20] Despite being one of the main victors in World War I, Italy entered a period of economic crisis and social turmoil, leading to the rise of a fascist dictatorship in 1922. Participation in World War II on the Axis side ended in military defeat, economic destruction and the Italian Civil War. Following the liberation of Italy and the rise of the resistance, the country abolished the monarchy, reinstated democracy, enjoyed a prolonged economic boom and, despite periods of sociopolitical turmoils, became a major advanced country.[21][22][23] Today, Italy has the third largest nominal GDP in the Eurozone and the eighth largest in the world. As an advanced economy, the country has the sixth-largest worldwide national wealth, and it is ranked third for its central bank gold reserve. Italy has a very high level of human development, and it stands among the top countries for life expectancy. The country plays a prominent role in regional and global economic, military, cultural and diplomatic affairs, and it is both a regional power[24][25] and a great power.[26][27] Italy is a founding and leading member of the European Union and a member of numerous international institutions, including the UN, NATO, the OECD, the OSCE, the WTO, the G7, the G20, the Union for the Mediterranean, the Council of Europe, Uniting for Consensus, the Schengen Area and many more. As a reflection of its cultural wealth, Italy is home to 54 World Heritage Sites, the most in the world, and is the fifth-most visited country.",
#         "Brazil (Portuguese: Brasil [bɾaˈziw]),[nt 1] officially the Federative Republic of Brazil (Portuguese: República Federativa do Brasil, About this sound listen (help·info)),[10] is the largest country in both South America and Latin America. At 8.5 million square kilometers (3.2 million square miles)[11] and with over 208 million people, Brazil is the world's fifth-largest country by area and the sixth most populous. The capital is Brasília, and the most populated city is São Paulo. The federation is composed of the union of the Federal District, the 26 states, and the 5,570 municipalities. It is the largest country to have Portuguese as an official language and the only one in the Americas,[12][13] besides being one of the most multicultural and ethnically diverse nations, due to the strong immigration from various places in the world. Bounded by the Atlantic Ocean on the east, Brazil has a coastline of 7,491 kilometers (4,655 mi).[14] It borders all other South American countries except Ecuador and Chile and covers 47.3% of the continent's land area.[15] Its Amazon River basin includes a vast tropical forest, home to diverse wildlife, a variety of ecological systems, and extensive natural resources spanning numerous protected habitats.[14] This unique environmental heritage makes Brazil one of 17 megadiverse countries, and is the subject of significant global interest and debate regarding deforestation and environmental protection. Brazil was inhabited by numerous tribal nations prior to the landing in 1500 of explorer Pedro Álvares Cabral, who claimed the area for the Portuguese Empire. Brazil remained a Portuguese colony until 1808, when the capital of the empire was transferred from Lisbon to Rio de Janeiro. In 1815, the colony was elevated to the rank of kingdom upon the formation of the United Kingdom of Portugal, Brazil and the Algarves. Independence was achieved in 1822 with the creation of the Empire of Brazil, a unitary state governed under a constitutional monarchy and a parliamentary system. The ratification of the first constitution in 1824 led to the formation of a bicameral legislature, now called the National Congress. The country became a presidential republic in 1889 following a military coup d'état. An authoritarian military junta came to power in 1964 and ruled until 1985, after which civilian governance resumed. Brazil's current constitution, formulated in 1988, defines it as a democratic federal republic.[16] Due to its rich culture and history, the country ranks thirteen in the world by number of UNESCO World Heritage Sites.[17] Brazil has the eighth largest GDP in the world by both nominal and PPP measures (as of 2017).[18][19] The nation is one of the world's major breadbaskets, being the largest producer of coffee for the last 150 years.[20] It is classified as an upper-middle income economy by the World Bank[21] and a newly industrialized country,[22][23] which holds the largest share of global wealth in Latin America. As a regional and middle power,[24][25] the country has international recognition and influence, being also classified as an emerging global power[26] and a potential superpower by several analysts.[27][28][29] Brazil is a founding member of the United Nations, the G20, BRICS, Union of South American Nations, Mercosul, Organization of American States, Organization of Ibero-American States and the Community of Portuguese Language Countries. ",
#         "Bulgaria (/bʌlˈɡɛəriə, bʊl-/ (About this sound listen); Bulgarian: България, tr. Bǎlgariya), officially the Republic of Bulgaria (Bulgarian: Република България, tr. Republika Bǎlgariya, IPA: [rɛˈpublikɐ bɐɫˈɡarijɐ]), is a country in southeastern Europe. It is bordered by Romania to the north, Serbia and Macedonia to the west, Greece and Turkey to the south, and the Black Sea to the east. The capital and largest city is Sofia; other major cities are Plovdiv, Varna and Burgas. With a territory of 110,994 square kilometres (42,855 sq mi), Bulgaria is Europe's 16th-largest country. Organised prehistoric cultures began developing on current Bulgarian lands during the Neolithic period. Its ancient history saw the presence of the Thracians, Ancient Greeks, Persians, Celts, Romans and others. The emergence of a unified Bulgarian state dates back to the establishment of the First Bulgarian Empire in 681 AD, which dominated most of the Balkans and functioned as a cultural hub for Slavs during the Middle Ages. With the downfall of the Second Bulgarian Empire in 1396, its territories came under Ottoman rule for nearly five centuries. The Russo-Turkish War of 1877–78 led to the formation of the Third Bulgarian State. The following years saw several conflicts with its neighbours, which prompted Bulgaria to align with Germany in both world wars. In 1946 it became a one-party socialist state as part of the Soviet-led Eastern Bloc. In December 1989 the ruling Communist Party allowed multi-party elections, which subsequently led to Bulgaria's transition into a democracy and a market-based economy. Bulgaria's population of 7.2 million people is predominantly urbanised and mainly concentrated in the administrative centres of its 28 provinces. Most commercial and cultural activities are centred to the capital and largest city, Sofia as well as the second and the third largest cities - Plovdiv and Varna. The strongest sectors of the economy are heavy industry, power engineering, and agriculture, all of which rely on local natural resources. The country's current political structure dates to the adoption of a democratic constitution in 1991. Bulgaria is a unitary parliamentary republic with a high degree of political, administrative, and economic centralisation. It is a member of the European Union, NATO, and the Council of Europe; is a founding state of the Organization for Security and Co-operation in Europe (OSCE); and has taken a seat at the UN Security Council three times."
#         ]
#
#tests = ["Syria (Arabic: سوريا‎ Sūriyā), officially known as the Syrian Arab Republic (Arabic: الجمهورية العربية السورية‎ al-Jumhūrīyah al-ʻArabīyah as-Sūrīyah), is a country in Western Asia, bordering Lebanon and the Mediterranean Sea to the west, Turkey to the north, Iraq to the east, Jordan to the south, and Israel to the southwest. Syria's capital and largest city is Damascus. A country of fertile plains, high mountains, and deserts, Syria is home to diverse ethnic and religious groups, including Syrian Arabs, Greeks, Armenians, Assyrians, Kurds, Circassians,[8] Mandeans[9] and Turks. Religious groups include Sunnis, Christians, Alawites, Druze, Isma'ilis, Mandeans, Shiites, Salafis, Yazidis, and Jews. Sunni make up the largest religious group in Syria. Syria is an unitary republic consisting of 14 governorates and is the only country that politically espouses Ba'athism. It is a member of one international organization other than the United Nations, the Non-Aligned Movement; it has become suspended from the Arab League on November 2011[10] and the Organisation of Islamic Cooperation,[11] and self-suspended from the Union for the Mediterranean.[12] In English, the name Syria was formerly synonymous with the Levant (known in Arabic as al-Sham), while the modern state encompasses the sites of several ancient kingdoms and empires, including the Eblan civilization of the 3rd millennium BC. Its capital Damascus and largest city Aleppo are among the oldest continuously inhabited cities in the world.[13] In the Islamic era, Damascus was the seat of the Umayyad Caliphate and a provincial capital of the Mamluk Sultanate in Egypt. The modern Syrian state was established in mid-20th century after centuries of Ottoman and a brief period French mandate, and represented the largest Arab state to emerge from the formerly Ottoman-ruled Syrian provinces. It gained de-jure independence as a parliamentary republic on 24 October 1945, when Republic of Syria became a founding member of the United Nations, an act which legally ended the former French Mandate – although French troops did not leave the country until April 1946. The post-independence period was tumultuous, and a large number of military coups and coup attempts shook the country in the period 1949–71. In 1958, Syria entered a brief union with Egypt called the United Arab Republic, which was terminated by the 1961 Syrian coup d'état. The republic was renamed into the Arab Republic of Syria in late 1961 after December 1 constitutional referendum, and was increasingly unstable until the Ba'athist coup d'état, since which the Ba'ath Party has maintained its power. Syria was under Emergency Law from 1963 to 2011, effectively suspending most constitutional protections for citizens. Bashar al-Assad has been president since 2000 and was preceded by his father Hafez al-Assad,[14] who was in office from 1971 to 2000. Since March 2011, Syria has been embroiled in an armed conflict, with a number of countries in the region and beyond involved militarily or otherwise. As a result, a number of self-proclaimed political entities have emerged on Syrian territory, including the Syrian opposition, Rojava, Tahrir al-Sham and Islamic State of Iraq and the Levant. Syria is ranked last on the Global Peace Index, making it the most violent country in the world due to the war, although life continues normally for most of its citizens as of December 2017. The war caused 470,000 deaths (February 2016 SCPR estimate),[15] 7.6 million internally displaced people (July 2015 UNHCR estimate) and over 5 million refugees (July 2017 registered by UNHCR),[16] making population assessment difficult in recent years."
#        "Nigeria (/naɪˈdʒɪəriə/ (About this sound listen)), officially the Federal Republic of Nigeria, is a federal republic in West Africa, bordering Benin in the west, Chad and Cameroon in the east, and Niger in the north. Its coast in the south lies on the Gulf of Guinea in the Atlantic Ocean. It comprises 36 states and the Federal Capital Territory, where the capital, Abuja is located. Nigeria is officially a democratic secular country.[6] Nigeria has been home to a number of kingdoms and tribal states over the millennia. The modern state originated from British colonial rule beginning in the 19th century, and took its present territorial shape with the merging of the Southern Nigeria Protectorate and Northern Nigeria Protectorate in 1914. The British set up administrative and legal structures whilst practising indirect rule through traditional chiefdoms. Nigeria became a formally independent federation in 1960. It experienced a civil war from 1967 to 1970. It thereafter alternated between democratically elected civilian governments and military dictatorships until it achieved a stable democracy in 1999, with the 2011 presidential election considered the first to be reasonably free and fair.[7] Nigeria is often referred to as the Giant of Africa, owing to its large population and economy.[8] With 186 million inhabitants, Nigeria is the most populous country in Africa and the seventh most populous country in the world. Nigeria has the third-largest youth population in the world, after India and China, with more than 90 million of its population under age 18.[9][10] The country is viewed as a multinational state as it is inhabited by over 500 ethnic groups, of which the three largest are the Hausa, Igbo and Yoruba; these ethnic groups speak over 500 different languages and are identified with a wide variety of cultures.[11][12] The official language is English. Nigeria is divided roughly in half between Christians, who live mostly in the southern part of the country, and Muslims, who live mostly in the north. A minority of the population practise religions indigenous to Nigeria, such as those native to the Igbo and Yoruba ethnicities. As of 2015, Nigeria is the world's 20th largest economy, worth more than $500 billion and $1 trillion in terms of nominal GDP and purchasing power parity respectively. It overtook South Africa to become Africa's largest economy in 2014.[13][14] The 2013 debt-to-GDP ratio was 11 percent.[15] Nigeria is considered to be an emerging market by the World Bank;[16] it has been identified as a regional power on the African continent,[17][18][19] a middle power in international affairs,[20][21][22][23] and has also been identified as an emerging global power.[24][25][26] However, it currently has a low Human Development Index, ranking 152nd in the world. Nigeria is a member of the MINT group of countries, which are widely seen as the globe's next BRIC-like economies. It is also listed among the Next Eleven economies set to become among the biggest in the world. Nigeria is a founding member of the African Union and a member of many other international organizations, including the United Nations, the Commonwealth of Nations and OPEC." 
#        "France (French: [fʁɑ̃s]), officially the French Republic (French: République française [ʁepyblik fʁɑ̃sɛz]), is a sovereign state whose territory consists of metropolitan France in Western Europe, as well as several overseas regions and territories.[XIII] The metropolitan area of France extends from the Mediterranean Sea to the English Channel and the North Sea, and from the Rhine to the Atlantic Ocean. The overseas territories include French Guiana in South America and several islands in the Atlantic, Pacific and Indian oceans. The country's 18 integral regions (five of which are situated overseas) span a combined area of 643,801 square kilometres (248,573 sq mi) and a total population of 67.25 million (as of June 2018).[10] France is a unitary semi-presidential republic with its capital in Paris, the country's largest city and main cultural and commercial centre. Other major urban centres include Marseille, Lyon, Lille, Nice, Toulouse and Strasbourg. During the Iron Age, what is now metropolitan France was inhabited by the Gauls, a Celtic people. Rome annexed the area in 51 BC, holding it until the arrival of Germanic Franks in 476, who formed the Kingdom of France. France emerged as a major European power in the Late Middle Ages following its victory in the Hundred Years' War (1337 to 1453). During the Renaissance, French culture flourished and a global colonial empire was established, which by the 20th century would be the second largest in the world.[11] The 16th century was dominated by religious civil wars between Catholics and Protestants (Huguenots). France became Europe's dominant cultural, political, and military power under Louis XIV.[12] In the late 18th century, the French Revolution overthrew the absolute monarchy, established one of modern history's earliest republics, and saw the drafting of the Declaration of the Rights of Man and of the Citizen, which expresses the nation's ideals to this day. In the 19th century Napoleon took power and established the First French Empire. His subsequent Napoleonic Wars shaped the course of continental Europe. Following the collapse of the Empire, France endured a tumultuous succession of governments culminating with the establishment of the French Third Republic in 1870. France was a major participant in World War I, from which it emerged victorious, and was one of the Allied Powers in World War II, but came under occupation by the Axis powers in 1940. Following liberation in 1944, a Fourth Republic was established and later dissolved in the course of the Algerian War. The Fifth Republic, led by Charles de Gaulle, was formed in 1958 and remains today. Algeria and nearly all the other colonies became independent in the 1960s and typically retained close economic and military connections with France. France has long been a global centre of art, science, and philosophy. It hosts Europe's third-largest number of cultural UNESCO World Heritage Sites and leads the world in tourism, receiving around 83 million foreign visitors annually.[13] France is a developed country with the world's seventh-largest economy by nominal GDP,[14] and ninth-largest by purchasing power parity.[15] In terms of aggregate household wealth, it ranks fourth in the world.[16] France performs well in international rankings of education, health care, life expectancy, and human development.[17][18] France is globally considered a great power in the world,[19] being one of the five permanent members of the United Nations Security Council with the power to veto and is an official nuclear-weapon state. It is a leading member state of the European Union and the Eurozone.[20] It is also a member of the Group of 7, North Atlantic Treaty Organization (NATO), Organisation for Economic Co-operation and Development (OECD), the World Trade Organization (WTO), and La Francophonie."
#        ]
#
#b = InsiderNLP(texts)
#b.clean()
#print(b.analyzeFrequencies())
#print(b.getFrequencies(1, 10, plot = True))
##b.doNER()
#b.analyzeSentiment()
#b.doLDA(True)
#b.getTopicLDA(1)
#b.getAllTopics('lda')
#b.doNMF(True)
#b.getTopicNMF(1)
#b.getAllTopics('nmf')
## test the predict methods
#c = InsiderNLP(titles)
#c.clean()
#c.predictTopicLDA()
#c.predictTopicNMF()
#c.getAllTopics('lda')
#c.getAllTopics('nmf')
#c.getTopicLDA(1)
#c.getTopicNMF(1)