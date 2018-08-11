# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:47:28 2018

@author: c14238a
"""

''' Currently, this doesn't have any particular idea, except for downloading
the articles for a given period and region and saving them to an excel file
'''
# Set the working directory
import os
#import inspect
#filename = inspect.getframeinfo(inspect.currentframe()).filename
#path = os.path.dirname(os.path.abspath(filename))
#path = path + '\\'
#print(path)
path = "D://Книги//Мои//Python//Articles//Articles//"
os.chdir(path)


path_mod = path + 'ModellingModule\\'
path_col = path + 'CollectionModule\\'

from CollectionModule.SpecificNewspapers import insidernewspaper
#from ModellingModule import implemented

import pandas as pd
import numpy as np


#url = 'https://www.insidermedia.com/insider'
#url_pattern = ('a', {'class':'t-none'})
#title_pattern = ('h1', {'itemprop':'headline'})
#text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
#date_pattern = ('time', {'itemprop':'datePublished'}, {'property':'datetime', 'format':'%Y-%m-%dT%H:%M'})
#topic_pattern = ('h1', {'itemprop':'headline'})
#region_pattern = ('span', {'itemprop':'contentLocation'})
#article_patterns = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern, 'topic':topic_pattern, 'region':region_pattern}
#page_not_found_texts = ['page not found', 'Page not found']
#nonexistent_page_patterns = ['There are no articles that match your search. Please try with different criteria.']

return_date_format = '%d %B %Y'

from_date = '29 June 2018' # Try only one day, works
to_date = '29 June 2018'  # Until the most recent point

regions = ['central-and-east', 'ireland', 'midlands', 'northeast', 'northwest', 
          'southeast', 'southwest', 'wales', 'yorkshire']
elements = ['date', 'title', 'text']


regions = ['yorkshire']
for region in regions:
    p = insidernewspaper.InsiderNewspaper(region, from_date = from_date, to_date = to_date)
    p.generateNewspages() 
    p.extractArticlesFromNewspages()

    final_df = pd.DataFrame(columns = ['date', 'location', 'topic', 'title', 'url', 'text'],
                            index = np.arange(len(p.getArticles())))
    
    final_df.date = p.getArticleElement('date', date_format = return_date_format)
    final_df.location = p.getArticleElement('region')
    final_df.topic = p.getArticleElement('topic')
    final_df.title = p.getArticleElement('title')
    final_df.text = p.getArticleElement('text')
    
    article_urls = []
    for newspage in p.getNewspages():
        article_urls.extend(newspage.getURLs())
        
    final_df.url = article_urls
    
#    final_df.to_excel('DownloadedData\\' + region + '.xlsx')