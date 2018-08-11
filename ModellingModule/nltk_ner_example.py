# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:51:47 2018

@author: Slav
"""

import nltk 
from nltk.tag import StanfordNERTagger

with open('D:\Книги\Мои\Python\Articles\Articles\\sampleCH.txt', 'r') as f:
    sample = f.read()


sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=False)

#import os
#java_path = "C:\\Program Files\\Java\\jdk1.8.0_162\\bin\\java.exe"
#os.environ['JAVAHOME'] = java_path
#nltk.internals.config_java("C:\\Program Files\\Java\\jdk1.8.0_162\\bin\\java.exe")
#st = StanfordNERTagger('D:\\Книги\\Мои\\Python\\Articles\\Articles\\ModellingModule\\StanfordNER\\classifiers\\english.muc.7class.distsim.crf.ser.gz', 
#                       'D:\\Книги\\Мои\\Python\\Articles\\Articles\\ModellingModule\\StanfordNER\\stanford-ner.jar', encoding = 'utf8')

#classified_sentences = [st.tag(sentence) for sentence in tokenized_sentences]


for chunked_sentence in chunked_sentences:
    for chunk in chunked_sentence:
        if hasattr(chunk, 'label'):
            print(chunk.label(), ' '.join(c[0] for c in chunk))

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

entity_names = []
for tree in chunked_sentences:
    # Print results per sentence
    # print extract_entity_names(tree)

    entity_names.extend(extract_entity_names(tree))

# Print all entity names
#print entity_names

# Print unique entity names
print(set(entity_names))