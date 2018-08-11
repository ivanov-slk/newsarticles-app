# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:09:28 2018

@author: c14238a

!!!
Mind that there are links to the specific InsiderNewspage and InsiderArticles in 
the GeneralNewspaper. 
!!!
"""
## Set the working directory
#import os
#path = "C://Work//Python//Articles//CollectionModule//"
##path = "D://Книги//Мои//Python//Articles//"
#os.chdir(path)

from CollectionModule.newspaper import *

class InsiderNewspaper(GeneralNewspaper):
    '''
    This class has the same purpose of GeneralNewspaper with the only difference
    (for now) that it has its settings (HTML tag patterns) saved as class variables.
    
    Refer to GeneralNewspaper for more information.
    '''
    url = 'https://www.insidermedia.com/insider'
    url_pattern = ('a', {'class':'t-none'})
    title_pattern = ('h1', {'itemprop':'headline'})
    text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
    date_pattern = ('time', {'itemprop':'datePublished'}, {'property':'datetime', 'format':'%Y-%m-%dT%H:%M'})
    topic_pattern = ('h1', {'itemprop':'headline'})
    region_pattern = ('span', {'itemprop':'contentLocation'})
    article_patterns = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern, 'topic':topic_pattern, 'region':region_pattern}
    page_not_found_texts = ['page not found', 'Page not found']
    nonexistent_page_patterns = ['There are no articles that match your search. Please try with different criteria.']
    
    def __init__(self, region, from_date = '', to_date = '' , auto_download = False, 
                 date_format = '%d %B %Y'):
        '''
        Refer to GeneralNewspaper. Here, the constant settings (HTML tag specifications,
        etc.) have been left as class variables.
        
        Parameters:
            region: string; a region's name as per insidermedia's url codes
            
            from_date: string; valid date in format dd mmmm yyyy
            
            to_date: string; valid date in format dd mmmm yyyy
            
            auto_download: boolean; True to automatically download the HTML codes
            
            date_format: string, represents a valid datetime library format. This 
                        should be the format in which the website dates are shown.
        '''
        self.region = region
        self.auto_download = auto_download
        self.date_format = date_format
        
        self.accessed_urls = []  # Here the accessed Newspage urls will be stored
        self.newspages = []
        
        # Instantiate the dates
        if from_date == '':
            self.from_date = datetime.datetime.strptime('01 January 1970', '%d %B %Y')
        else:
            self.from_date = datetime.datetime.strptime(from_date, self.date_format)
        
        if to_date == '':
            self.to_date = datetime.datetime.now()
        else:
            self.to_date = datetime.datetime.strptime(to_date, self.date_format)


##### Tests
#url = 'https://www.insidermedia.com/insider'
region = 'wales'
#url_pattern = ('a', {'class':'t-none'})
#title_pattern = ('h1', {'itemprop':'headline'})
#text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
#date_pattern = ('time', {'itemprop':'datePublished'}, {'property':'datetime', 'format':'%Y-%m-%dT%H:%M'})
#topic_pattern = ('h1', {'itemprop':'headline'})
#region_pattern = ('span', {'itemprop':'contentLocation'})
#article_patterns = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern, 'topic':topic_pattern, 'region':region_pattern}
#page_not_found_texts = ['page not found', 'Page not found']
#nonexistent_page_patterns = ['There are no articles that match your search. Please try with different criteria.']

#from_date = '31 December 2017'
#to_date = '1 January 2018'
#p = InsiderNewspaper(url, region, url_pattern, article_patterns, 
#                 page_not_found_texts, from_date = from_date, to_date = to_date)
#p.generateNewspages() # Raises the exception in the Newspage method - as expected - there are no articles for this period
#
#from_date = '21 Dec 2017' #This is boundary condition, works
#to_date = '5 Jan 2018'
#from_date = '29 Apr 2016' # Check two different months, works
#to_date = '6 May 2016'
#from_date = '28 Oct 2017' # Check two different months, where there is no article from these days. works
#to_date = '6 Nov 2017'
#date_format = '%d %b %Y'
#p = InsiderNewspaper(url, region, url_pattern, article_patterns, nonexistent_page_patterns,
#                 page_not_found_texts, from_date = from_date, to_date = to_date, date_format = date_format)
#p.generateNewspages()  # seems ok
#p.extractArticlesFromNewspages()
#
#from_date = '16 May 2018' # Try only one day, works
#to_date = '17 May 2018'
#from_date = '29 June 2018' # Try only one day, works
#to_date = ''  # Until the most recent point
#p = InsiderNewspaper(region, from_date = from_date, to_date = to_date)
#p.generateNewspages() 
#p.extractArticlesFromNewspages()  