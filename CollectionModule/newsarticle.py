# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:25:11 2018

@author: c14238a
"""
from CollectionModule.articles_interfaces import *
from CollectionModule.exceptions import *

class NewsArticle(Article):
    '''
    This is an implemented version of the Article interface.
    
    It saves as an object attribute only the downloaded raw HTML. Any elements
    that may be extracted are returned and NOT saved as object attributes.
    
    For more information, please refer to the Article class docstring.
    '''
    def __init__(self, url, page_not_found_texts = ["Page not found"], 
                 auto_download = False):
        ''' The constructor method, obviously. The only argument it accepts is 
        the URL of the news article, which it saves as an object attribute.
        It also instantiates all other attributes:
            extracted_elements: a dictionary, keys: the names of the elements 
            (i.e. 'title', 'date'); values: the actual values
            extraction_pattern: a dictionary, keys: the names of the elements;
            values: two-element tuples containing 1. the HTML tag and 2. its 
            properties/attributes
            
        Parameters:
            page_not_found_texts: list; contains strings with the possible error
                            messages in case the URL is not found on the website.
            auto_download: boolean; True to download the raw HTML automatically.
        Returns:
            None; instantiates the object and its attributes.
        '''
        # Save the URL to a object attribute and initialize the other attributes
        self.url = url
        self.raw = ''
        self.extracted_elements = {'keep_flag':0} # This is going to be overwritten,
        # but the keep_flag is added in the extraction method in temp_dict, which
        # is going to overwrite self.extracted elements. It should be fine.
        self.extraction_patterns = {}
        self.page_not_found_texts = page_not_found_texts
        
        # Auto-download if specified by user
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
        # Check if self.url exists
        temp = getattr(self, 'url', None)
        if temp == None:
            raise AttributeError("The article URL address could not be found. " +
                                 "Please check if it has been successfully supplied " +
                                 "to the class constructor.")
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
            
    def getExtractedElement(self, element_name, date_format = '%d %B %Y'):
        '''
        Returns an already extracted element from self.extracted_elements. Raises
        KeyError if the requested element is not in the dictionary
        
        Parameters:
            element_name: string, should be one of the names of the extracted elements
        
        Returns:
            the contents of an HTML tag, usually a string.
        '''
        # Try returning the requested element
        try:
            # Check if a date is requested and format it as datetime.datetime
            if element_name == 'date':
#                if date_format == '':
#                    return self.extracted_elements[element_name]
#                return datetime.datetime.strftime(self.extracted_elements[element_name], date_format)
                # The above three lines were providing a easy-to-read date when 
                # requested, but as this method is not intended for user interaction,
                # They have been commented. Now, if the user requests the date of
                # the article, a datetime.datetime object will be provided (making
                # the `if` part kind of useless...)
                return self.extracted_elements['date']
            elif element_name == 'text':
                return self.extracted_elements[element_name].strip('\n')#.replace('\n', ' ')
            else:
                return self.extracted_elements[element_name]
        # Raise KeyError if unsuccessful
        except KeyError:
            raise KeyError("It seems that 'element_name' is not already extracted.")
            
    def getExtractionPatterns(self):
        '''
        Returns the HTML code patterns used to extract the requested elements 
        (the method extractAll should have been called prior to invoking this one)
        '''
        return self.extraction_patterns
        
    def extractElement(self, tag_pattern):
        '''
        This method extracts and returns the contents of the HTML tag specified 
        by tag_pattern.
        
        Parameters:
            tag_pattern: a tuple; the first element is the HTML tag as string
                the second element is the HTML tag properties as a dictionary
        Returns:
            string; the contents of the HTML tag specified by tag_pattern
        '''    
        ## This is kind of useless (since self.raw is instantiated), but may serve 
        ## as a code reference
        # Check if self.raw exists
        temp = getattr(self, 'raw', None)
        if temp is not None:
            # If exists try to find the value of tag_pattern
            try:
                return self.raw.find(tag_pattern[0], tag_pattern[1]).text
            # If there isn't such a value, raise an error (the .find() will return None)
            except AttributeError:
                raise NotFoundError("It seems that the tag_pattern you provided doesn't" \
                                + "exist in the raw HTML." + "\n" + self.url + "\n\n")
                
        # Handle the case if self.raw doesn't exist.
        else:
            raise DownloadError("Please invoke the downloadHTML method first.")
    
    def extractDate(self, tag_pattern):
        '''
        This method extracts a Date specifically from an article - insidermedia.com's
        full dates are stored differently than the title or text.
        Parameters:
            tag_pattern: string, the HTML pattern, where the element of interest
            is in.
        Returns:
            datetime object, the date of publishing of the article.,
        '''
        # Find the date of publishing tag element
        date_tag = self.raw.find(tag_pattern[0], tag_pattern[1])
        # Get the date and convert to datetime
        date_time = date_tag.get(tag_pattern[2]['property'])
        # Format date_time
        date_time = datetime.datetime.strptime(date_time, tag_pattern[2]['format'])
        # Return
        return date_time
        
    def extractTopic(self, tag_pattern):
        '''
        This method extract a Topic specifically from an article - insidermedia.com's
        topics are strangely encoded in HTML, so the tag in the headline should do the trick.
        '''
        # Extract the headline
        headline_tag = self.raw.find(tag_pattern[0], tag_pattern[1])
        # Get the headline data-categories property contents
        data_categories = headline_tag.get('data-categories')
        # Extract the topic
        ## This is a quick fix in case there is no topic
        try:
            return data_categories.split(',')[1]
        except IndexError:
            return ''
    
    def extractAll(self, tag_pattern_dictionary):
        '''
        This method extracts all requested elements (the keys of tag_pattern)
        from the downloaded raw HTML and saves them in the respective individual
        article attribute (a dictionary itself)
        
        Parameters:
            tag_pattern_dictionary: a dictionary; keys are the names of the ele-
            ments-to-be-extracted (i.e. 'title'), the values is a two-element
            tuple consisting of 1. the HTML tag name (string) and 2. the HTML
            tag attributes
            
        Returns:
            None, updates self.extracted_elements and self.extraction_patterns
        '''
        # Instantiate a temporary dictionary to store the extracted elements
        temp_dict = {'keep_flag':0}
        # Loop over the keys of tag_pattern_dictionary
        for key, value in tag_pattern_dictionary.items():
            # If the key is date, invoke extractDate
            if key == 'date':
                temp_dict[key] = self.extractDate(value)
            elif key == 'topic':
                temp_dict[key] = self.extractTopic(value)
            else:
                # Add the key-value pair to the temporary dictionary (call extractElement)
                temp_dict[key] = self.extractElement(value)
        # Update self.extracted_elements
        self.extracted_elements = temp_dict
        self.extraction_patterns = tag_pattern_dictionary
        
    def __lt__(self, other):
        '''
        Defines the sorting rule for objects of the class Article. The natural
        rule is to sort by date, although the implementations may differ.
        '''
        try:
            return self.extracted_elements['date'] < other.extracted_elements['date']
        except KeyError:
            print('Date is not extracted, sorting by titles.')
            return self.extracted_elements['title'] < other.extracted_elements['title']
        
 
    
    
    


###### NewsArticle tests
#url = "https://www.insidermedia.com/insider/northwest/major-energy-project-to-support-north-west-job-creatio"
#url1 = 'https://www.insidermedia.com/insider/northwest/manchester-tech-incubator-launches'
#a = NewsArticle(url)
#b = NewsArticle(url1)
#title_pattern = ('h1', {'itemprop':'headline'})
#text_pattern = ('div', {'class':'article-content', 'itemprop':'articleBody'})
#date_pattern = ('time', {'itemprop':'datePublished'}, {'property':'datetime', 'format':'%Y-%m-%dT%H:%M'})
#pattern_dictionary = {'title':title_pattern, 'text':text_pattern, 'date':date_pattern}
#
#a.downloadHTML()
#b.downloadHTML()
#b.extractAll(pattern_dictionary)
#b.getExtractedElement('date')
#
#patterns = [title_pattern, text_pattern, date_pattern]
#for pattern in patterns:    
#    print(a.extractElement(pattern))
#    print()
#    print(b.extractElement(pattern))
#    print()
#
## test invalid url
#url2 = "https://www.insidermedia.com/insider/northwest/major-energy-project-to-support-north-west-job-creatio"
#c = NewsArticle(url2)
#c.downloadHTML() # It just downloads the error 404 HTML
#c.extractElement(title_pattern) # should return NotFoundError
#
## test invalid pattern
#invalid_pattern = ('h1', {'itemprop':'headlin'})
#a.extractElement(invalid_pattern) # should return NotFoundError
## So far - passed
#######