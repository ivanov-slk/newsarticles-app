# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:09:28 2018

@author: c14238a
"""
# Set the working directory
import os
path = "C://Work//Python//"
#path = "D://Книги//Мои//Python//Articles//"
os.chdir(path)

from newspage import *

SLEEP_TIME = 3

class InsiderNewspaper(Newspaper):
    ''' 
    A Newspaper is a collection of Newspages, providing an abstraction over the
    Newspages' data (articles included) and provides an interface for working with
    these data.
    
    Formally, it should be sort of a collection of urls, each of which can be 
    interpreted as a Newspage, i.e. from it a Newspage object can be constructed and
    the latter's methods can be properly applied.
    
    Then, the class would initialize in itself various Newspages from the URL-s,
    which on their part would initialize the relevant Articles.
    
    '''
    
    def __init__(self, url, region, url_pattern, article_patterns, 
                 page_not_found_texts = ["Page not found"], auto_download = False,
                 from_date = '', to_date = '', date_format = '%d %B %Y'):
        '''
        Creates an object instance and instantiates all object attributes (including
        the arguments needed for the Newspage and Article classes).
        
        Parameters:
            url: string; valid URL address containing other url-s that can be
            interpreted as Newspages
            url_pattern: a tuple of two elements; the first element is a string 
                        of the name of the HTML tag that contains all relevant
                        article URL-s; the second element is a dictionary with 
                        the tag's properties.
            article_patterns: a dictionary; keys represent strings with the elements
                            that should be extracted (i.e. 'title', 'date', etc.);
                            the value is a tuple analogous to the url_pattern.
            page_not_found_texts: list; contains strings with the possible error
                            messages in case the URL is not found on the website.
            from_date: string; date format: '%d %B %Y'. The date since which articles
                       will be extracted
            to_date: string; date format: '%d %B %Y'. The date up to which articles
                     will be extracted
            date_format: string, represents a valid datetime library format. This 
                        should be the format in which the website dates are shown.
            
        Returns:
            None; instantiates the object.
        '''
        self.url = url
        self.region = region
        self.url_pattern = url_pattern
        self.article_patterns = article_patterns
        self.page_not_found_texts = page_not_found_texts
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
            
    
    def generateURLs(self):
        '''
        Generates the URL-s of the individual Newspages as appropriate.
        This method is currently adapted to insidermedia.com's address structure.
        '''
        # Initialize a counter variable
        i = 0
        # Loop 1000 times
        for i in range(1000):
            # Construct the url address
            newspage_url = self.url + '/' + self.region + '/P' + str(i) + '0?year=' \
                            + str(self.from_date.year) + '&year=' + str(self.to_date.year)
            # Yield
            yield newspage_url
    
    def generateNewspages(self):
        '''
        Creates a Newspage object from each URL. 
        
        It might also invoke the do_all() method of each Newspage, or calls 
        sequentially the do_all methods from Newspage - depends on whether do_all
        is implemented in all Newspages.
        
        Mind the stop_extracting variable from the Newspage class.
        '''
        # Initialize a list where the  Newspages will be stored
        newspages = []
        # Initialize the stop_extracting variable
        stop_extracting = False
        # Initialize the URL generator
        url_generator = self.generateURLs()
        # Loop while stop_extracting = False or the counter is less than 1000 (to be sure that it will stop at some point)
        while stop_extracting == False:
            # Get new url
            url_address = next(url_generator)
            # Add the url address to the list of accessed urls
            self.accessed_urls.append(url_address)
            # Initialize a new Newspage
            temp_newspage = InsiderNewspage(url = url_address, url_pattern = self.url_pattern, 
                                            article_patterns = self.article_patterns, 
                                            page_not_found_texts = self.page_not_found_texts,
                                            from_date = self.from_date, 
                                            to_date = self.to_date, 
                                            date_format = self.date_format)
            temp_newspage.downloadHTML()
            stop_extracting = temp_newspage.extractURLs()
            # Add to the list
            self.newspages.append(temp_newspage)
        
        
    def extractArticlesFromNewspages(self):
        '''
        This method calls initializeArticles and extractAll methods of all newspages
        '''
        for newspage in self.newspages:
            newspage.initializeArticles(True)
            newspage.extractAllFromArticles()
        
    def getURLs(self):
        '''
        Returns all generated Newspage URL-s. 
        '''
        return self.accessed_urls
        
    def getNewspages(self):
        '''
        Returns all Newspages.
        '''
        return self.newspages
    
    def getArticles(self):
        '''
        Returns all articles from all newspages.
        '''
        # Initialize the final list
        all_articles = []
        # Loop over all newspages
        for newspage in self.newspages:
            # Extract all articles from a single newspage
            extracted_articles = newspage.getArticles()
            # If there are any articles, extend the final list
            if extracted_articles != []:
                all_articles.extend(extracted_articles)
            
        return all_articles
    
    def getArticleElement(self, element_name):
        '''
        Returns a specified element name from all articles from all newspages.
        Parameters:
            element_name: string, valid article element
        Returns:
            the article element; raises NotFoundError if the elment is not found.
        '''
        # Initialize the final list
        all_elements = []
        # Loop over all newspages
        for newspage in self.newspages:
            # Extract all elements from a single newspage
            extracted_elements = newspage.getElementFromAllArticles(element_name)
            # If there are any elements, extend the final list (extend as we don't want nested lists)
            if extracted_elements != []:
                all_elements.extend(extracted_elements)
            
        return all_elements
        
        
##### Tests
url = 'https://www.insidermedia.com/insider/'
region = 'northeast'
url_pattern = ('a', {'class':'t-none'})
title_pattern = ('h1', {'itemprop':'headline'})
text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
date_pattern = ('time', {'itemprop':'datePublished'})
article_patterns = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern}
page_not_found_texts = ['page not found', 'Page not found']

#from_date = '31 December 2017'
#to_date = '1 January 2018'
#p = InsiderNewspaper(url, region, url_pattern, article_patterns, 
#                 page_not_found_texts, from_date = from_date, to_date = to_date)
#p.generateNewspages() # Raises the exception in the Newspage method - as expected - there are no articles for this period

from_date = '21 Dec 2017'
to_date = '5 Jan 2018'
date_format = '%d %b %Y'
p = InsiderNewspaper(url, region, url_pattern, article_patterns, 
                 page_not_found_texts, from_date = from_date, to_date = to_date, date_format = date_format)
#p.generateNewspages()  # seems ok
#p.extractArticlesFromNewspages()

#from_date = '10 May 2018'
#to_date = '10 May 2018'
#p = InsiderNewspaper(url, region, url_pattern, article_patterns, 
#                 page_not_found_texts, from_date = from_date, to_date = to_date)
#p.generateNewspages() # Seems ok
#p.extractArticlesFromNewspages()  # Seems ok