# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:09:28 2018

@author: c14238a
"""
## Set the working directory
#import os
#path = "C://Work//Python//Articles//CollectionModule//"
#path = "D://Книги//Мои//Python//Articles//Articles//"
#os.chdir(path)

from CollectionModule.newspage import *

SLEEP_TIME = 3

class GeneralNewspaper(Newspaper):
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
                 nonexistent_page_patterns,
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
            nonexistent_page_patterns: list of strings containing this website's
                            specific nonexistent page messages
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
        self.nonexistent_page_patterns = nonexistent_page_patterns
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
        # Initialize a list with the url addresses
        url_addresses = []
        # Calculate the difference in months between the starting and the ending dates
        month_difference = dateutil.relativedelta.relativedelta(self.to_date, self.from_date)        
        ### If month_difference is zero, check if both dates are in the same month
        if month_difference == 0:
            # If so, then we have to search two different months, and url_addresses should have two elements, each of which will use the RESPECTIVE data in self.to_date and self.from_date
            if self.to_date.month != self.from_date.month:
                url_addresses.append(self._makeURL(month = self.to_date.month, 
                                                   year = self.to_date.year))
                url_addresses.append(self._makeURL(month = self.from_date.month,
                                                   year = self.to_date.year))
            elif self.to_date.month == self.from_date.month:  # It shouldn't matter whether we use to_date of from_date
                url_addresses.append(self._makeURL(month = self.to_date.month,
                                                   year = self.to_date.year))
        ### If the difference is 1 or larger then generate a list of datetimes
        else:  
        # Create a list of all months between self.to_date and self.from_date
            dates_list = [dt for dt in dateutil.rrule.rrule(dateutil.rrule.MONTHLY, 
                                                        dtstart = self.from_date,
                                                        until = self.to_date)
                        ]
        # If the last month in dates_list is NOT the same as self.to_date, 
        # add the last month to the list of datetimes
        # It is in a while loop, just in case - this shouldn't make issues, but it might be worth checking.
        # Otherwise, a simple if statement should do the trick.
        # Besides, perhaps the while loop might do the job the list comprehension above does.
            while dates_list[-1].month != self.to_date.month:
                dates_list.append(dates_list[-1] + dateutil.relativedelta.relativedelta(months = 1))
        
        # Loop over dates and create url-s from them
            for a_date in dates_list:
                url_addresses.append(self._makeURL(month = a_date.month, year = a_date.year))
        
        ### Now the months and years should be ok and now a list with the url addresses that
        ### contain the search filters (i.e. month and year), continue with the page settings.
        return url_addresses
            
    
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
        # Get the relevant url addresses, given the specified dates
        url_addresses_filtered = self.generateURLs()
        # Loop over them and for each initialize a generator to loop over the pages
        for filtering_url in url_addresses_filtered:
            # Initialize the stop_extracting variable - it needs to be within the for loop
            stop_extracting = False
            # Initialize the page generator for each filtering url address
            page_generator = self._loopOverPages(filtering_url)
            # Loop while stop_extracting = False or the counter is less than 1000 (to be sure that it will stop at some point)
            while stop_extracting == False:
                # Get new url
                url_address = next(page_generator)
                print()
                print()
                print(url_address)
                print()
                print()
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
                ### Check if temp_newspage contains non-existent webpage. Perhaps
                ### implement a general method that does this inside the Newspaper class?
                ### The method should be generally applicable (not restricted to insidermedia.com).
                ### Specify a pattern for empty page and let the method check for it.
                ### If it is indeed blank then just don't include it in self.newspages.
                if self.checkNewspageExistence(temp_newspage) == True:
                    stop_extracting = temp_newspage.extractURLs()
                    # Add to the list
                    self.newspages.append(temp_newspage)
                else:
                    stop_extracting = True
                
        
        
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
    
    def getArticleElement(self, element_name, date_format = '%d %B %Y'):
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
            # Extract all elements from a single newspage (if element_name is 'url'
            # then invoke the getURLs() method of Newspage)
            if element_name == 'url':
                extracted_elements = newspage.getURLs()
            else:
                extracted_elements = newspage.getElementFromAllArticles(element_name, date_format = date_format)
            # If there are any elements, extend the final list (extend as we don't want nested lists)
            if extracted_elements != []:
                all_elements.extend(extracted_elements)
            
        return all_elements
    
    def setFromDate(self, from_date):
        '''
        Sets the from date anew. If the data methods are invoked, the existing
        articles/newspages will be overwritten.
        
        Parameters:
            from_date: string; valid date in format dd mmmm yyyy or datetime.datetime
        '''
        if type(from_date) == datetime.datetime:
            self.from_date = from_date
        elif type(from_date) == str:
            self.from_date = datetime.datetime.strptime(from_date, self.date_format)
        else:
            raise TypeError('Please provide the date either as a string in format dd mmmm yyyy or as a datetime.datetime')
            
    
    def setToDate(self, to_date):
        '''
        Sets the to date anew. If the data methods are invoked, the existing
        articles/newspages will be overwritten.
        
        Parameters:
            to_date: string; valid date in format dd mmmm yyyy
        '''
        if type(to_date) == datetime.datetime:
            self.to_date = to_date
        elif type(to_date) == str:
            self.to_date = datetime.datetime.strptime(to_date, self.date_format)
        else:
            raise TypeError('Please provide the date either as a string in format dd mmmm yyyy or as a datetime.datetime')
    
    def _makeURL(self, month, year, region = ''):
        '''
        Private method, used to save the space for writing a valid insidermedia
        URL address each time. It just plugs in month and year into the address pattern.
        Parameters:
            month: int between 1 and 12
            year: int
        Returns:
            string
        '''
        if region == '':
            region = self.region
        newspage_url = self.url + '/' + region + '/P{}' + '0?month=' +  \
                        str(month) + '&year=' + str(year)                     
        return newspage_url
        
    def _loopOverPages(self, url):
        '''
        Private method; loops over the pages of a url address on insidermedia.com that has already
        filters specified
        '''
        # Initialize a counter variable
        i = 0
        # Loop 1000 times
        for i in range(1000):
            # Construct the url address
            newspage_url = url.format(i)
            # Yield
            yield newspage_url
            
    def checkNewspageExistence(self, newspage_object):
        '''
        This method checks the contents of a Newspage for a pre-specified pattern
        that represents a non-existent page error or whatever the website shows when
        a non-existent page is accessed. 
        Parameters:
            newspage_object: a Newspage class instance; should have its HTML code downloaded.
        Returns:
            boolean; whether the url address is an existent page (True) or not (False)
        '''
        # Initialize a raw html variable
        raw_html = newspage_object.getRawHTML()
        # Check its contents and return
        for descendant in raw_html.descendants:
            for pattern in self.nonexistent_page_patterns:
                
                if pattern in descendant:
                    return False
        return True
