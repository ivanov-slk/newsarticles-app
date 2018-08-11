# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:28:00 2018

@author: c14238a
"""
# Set the working directory
#import os
#path = "C://Work//Python//"
#path = "D://Книги//Мои//Python//Articles//"
#os.chdir(path)

from CollectionModule.newsarticle import *

def find_nth(haystack, needle, n):
    ''' This function returns the index of the n-th occurence of a substring
    (needle) in a string (haystack). Credits: StackOverflow https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
    
    Parameters:
        haystack: a string
        needle: a string
        n:  an integer
    Returns:
        int
    ''' 
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

class InsiderNewspage(Newspage):
    '''    
    This is an implementation of the Newspage interface, adjusted for the 
    insidermedia.com's HTML code structure. 
    
    For more information about the Newspage interface, please refer to the 
    docstring of the interface.
    '''
    
    def __init__(self, url, url_pattern, article_patterns, 
                 page_not_found_texts = ["Page not found"], auto_download = False,
                 from_date = '', to_date = '', date_format = '%d %B %M'):
        '''
        The class constructor, obviously. It isn't intended to do anything per 
        se, except for initializing the Newspage instance and saving the arguments
        to an object attributes. It also infers the main website address from the
        url.
        
        Parameters:
            url: a string representing a valid URL address
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
            None, just instantiates the object.
        '''
        # Instantiate all object attributes
        self.url = url
        self.url_pattern = url_pattern
        self.article_patterns = article_patterns
        self.page_not_found_texts = page_not_found_texts
        
        self.raw = ''
        self.article_urls = []
        self.articles = []
        
        # Instantiate the dates
        if from_date == '':
            self.from_date = datetime.datetime.strptime('01 January 1970', '%d %B %Y')
        elif type(from_date) == datetime.datetime:  # In case the __init__ arguments are entered by another program
            self.from_date = from_date
        else:
            self.from_date = datetime.datetime.strptime(from_date, '%d %B %Y')
        
        if to_date == '':
            self.to_date = datetime.datetime.now()
        elif type(to_date) == datetime.datetime:  # In case the __init__ arguments are entered by another program
            self.to_date = to_date
        else:
            self.to_date = datetime.datetime.strptime(to_date, '%d %B %Y')
        
        
        
        # Infer the main website address from the url
        if url[:3] == 'www':
            end_slash_index = find_nth(self.url, '/', 1)
        elif url[:3] == 'htt':
            end_slash_index = find_nth(self.url, '/', 3)
        else:
            raise ValueError("Unrecognized url pattern. Please check the input url.")
        # Extract the main website address and store it to an attribute
        self.main_address = url[:(end_slash_index)]
        
        # Download automatically if requested
        if auto_download:
            self.downloadHTML()
    
    def downloadHTML(self):
        '''
        This method sends a request to the specified URL and "soupifies" the 
        result, saving it to an object attribute.
        
        Parameters:
            --
        Returns:
            None, updates the object attribute 'raw'
        '''
        # Make GET request
        response = requests.get(self.url, verify = False)
        # "Soupify" the result and add to an attribute
        self.raw = bs4.BeautifulSoup(response.text, 'html.parser')  
        # Do a check to see if error 404 has occured
        for descendant in self.raw.descendants:
            for text in self.page_not_found_texts:
                if text in descendant:
                    raise NotFoundError("It seems that an error 404 occured. Please check your URL-s.")
        # Check if raw is empty
        if self.raw == '':
            raise NotFoundError("No HTML code found. Please check your URL-s.")
        # Sleep, in order not to spam the website
        time.sleep(SLEEP_TIME)
        
    def extractURLs(self):
        '''
        This method extracts the article URL-s from the downloaded HTML code and
        stores them in an appropriate data structure.
        
        Note, the method may be initializing more variables than needed, but this
        helps readability.
        '''
        # Initialize the checking variable - if True then 
        stop_extracting = False
        # Initalize an empty list where the extracted tag objects will be stored
        raw_urls = []
        # Initialize an empty list where the url-s will be stored
        final_urls = []
        # Invoke the find_all method of the raw HTML, this returns a list of tag 
            # objects and store to the list
        raw_urls = self.raw.find_all(name=self.url_pattern[0], attrs=self.url_pattern[1])
        # Search the raw_urls to check if the program has tried to access a non-existent page
        
        # Extract the urls, add insider's base address and store to the final list
        for url in raw_urls:
            if self._extractDate(url) >= self.from_date and self._extractDate(url) <= self.to_date:
                final_urls.append(self.main_address + url.get('href'))
            if self._extractDate(url) < self.from_date:
                stop_extracting = True
#        final_urls = [self.main_address + url.get('href') for url in raw_urls if
#                      self._extractDate(url) >= self.from_date and self._extractDate(url) <= self.to_date]
        # Check if final_urls is not empty
        if final_urls == [] and self._extractDate(raw_urls[-1]) >= self.from_date and self._extractDate(raw_urls[0]) <= self.to_date:
            raise AttributeError("The URL-s couldn't be extracted. Please check your inputs.")
        # Store to an object attribute
        self.article_urls = final_urls
        # Return stop_extracting - this is important for the Newspaper
        return stop_extracting
        
    def initializeArticles(self, auto_download = False):
        '''
        This method uses the extracted by the above method URL-s and instantiates
        the respective Article objects.
        '''
        # Instantiate an empty list where the Article objects will be stored
        articles = []
        # Loop over self.article_urls and instantiate the articles
        for url in self.article_urls:
            articles.append(NewsArticle(url, self.page_not_found_texts, auto_download))
        
        # Store to an object attribute
        self.articles = articles
        
    def extractAllFromArticles(self):
        '''
        This method invokes Article.extractAll method for each article in self.articles.
        '''
        for article in self.articles:
            article.extractAll(self.article_patterns)
    
    def getRawHTML(self):
        '''
        A Getter method; returns the raw ("soupified") HTML code. If the 
        downloadHTML method has not been invoked (i.e. self.raw doesn't exist)
        the method raises an exception.
        '''
        # Check if downloadHTML has been invoked
        try:
            return self.raw
        # If not, raise an Exception
        except AttributeError:
            raise DownloadError("Please invoke the downloadHTML method first.")
        
    def getURLs(self):
        '''
        A Getter method; returns the extracted article URL-s.
        Again, consider handling of situations analogous to the above method
        '''
        return self.article_urls
        
    def getArticles(self):
        '''
        A Getter method; returns the Article objects that have been instantiated
        previously. Consider handling if initializeArticles() hasn't been invoked.
        '''
        return self.articles
    
    def getElementFromAllArticles(self, element_name, date_format = '%d %B %Y'):
        '''
        Returns a list with 'element_name' from all articles.
        
        Parameters:
            element_name: a string; valid extracted element's name
            
        Returns:
            list; contains the contents of each article's 'element_name'
        '''
        # Initialize the return list
        return_list = []
        # Loop over the articles and append to list
        for article in self.articles:
            return_list.append(article.getExtractedElement(element_name, date_format = date_format))
        # Return the list
        return return_list
    
    def do_all(self):
        ''' 
        This method calls self.downloadHTML(), self.extractURLs(), self.initia-
        lizeArticles() and self.extractAllFromArticles() successively.
        Parameters:
            None needed.
        Returns:
            None, updates the attributes of self
        ''' 
        self.downloadHTML()
        self.extractURLs()
        self.initializeArticles(True)
        self.extractAllFromArticles()
    
    def _extractDate(self, url):
        '''
        This method extracts the date from the container of the link to an article.
        It is needed to subset the URL-s based on pre-specified time period.
        The method is defined as private, formally - it is not intended to be invoked
        by the user.
        '''
        return datetime.datetime.strptime(url.parent.next_sibling.next_sibling.text.strip(), '%d %b %Y')
        
        
    def __iter__(self):
        ''' This is a generator to loop over the articles.
        Not quite sure if it should be kept this way - perhaps rename it, so it
        doesn't take a "reserved" name?
        ''' 
        pass
        for article in self.articles:
            yield article
    
    
###### Tests
#n_url = 'https://www.insidermedia.com/insider/northeast'
#url_pattern = ('a', {'class':'t-none'})
#title_pattern = ('h1', {'itemprop':'headline'})
#text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
#date_pattern = ('time', {'itemprop':'datePublished'})
#article_patterns = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern}
#page_not_found_texts = ['page not found', 'Page not found']

#n = InsiderNewspage(n_url, url_pattern, article_patterns, page_not_found_texts)
#n.downloadHTML()
#n.extractURLs()
#n.getURLs()
#n.initializeArticles(True)
#n.getArticles()
#n.extractAllFromArticles()
#n.do_all()
#n.getElementFromAllArticles('title')
#n.getElementFromAllArticles('date')
#n.getElementFromAllArticles('text')

