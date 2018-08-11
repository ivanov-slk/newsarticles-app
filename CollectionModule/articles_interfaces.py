# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:17:18 2018

@author: Slav

Some general notes:
The basic structure of the classes is as follows: the Article is the core structural
element. Newspage is a collection of Articles. Newspaper is a collection of Newspages.

I.e. Article --> Newspage --> Newspaper

In general, any class should use classes below it and NOT use classes above it
(i.e. Newspage may invoke Article methods, but not Newspaper methods)

Sort of a naming convention:
    class names follow CapWords convention;
    method names follow mixedWords convention;
    function or variable names follow lowercase_with_underscore convention;

Please note that this might not follow too closely the Style Guide for Python Code:
https://www.python.org/dev/peps/pep-0008/#id42
"""
### Articles and Newspages
import requests, datetime, bs4, time, dateutil

from abc import ABC, abstractmethod

SLEEP_TIME = 3  # Five seconds delay is ok. Don't want to spam the website, eh?

class Article(ABC):
    '''
    This is an Abstract Base Class for the class Article. It specifies the most
    important class attribute - the URL - and also the most important methods
    that an object of this class should have.
    This interface is not meant to be implemented. Its children could, however,
    add new methods or attributes to the class they inherit.
    
    An article is basically an HTML code of a webpage that is assumed to repre-
    sent a standard news article from a website. Its most important attribute
    is the URL, without which the object doesn't quite make sense. 

    The article object serves the purpose of 1. downloading and "soupifying" a 
    news article; 2. extracting elements from the HTML code based on a given
    HTML tag attributes. Various methods may be implented in the children classes
    to perform these tasks.
    
    Example methods (just to keep ideas written somewhere): 
        setAttributesDictionary: there may be a few methods of this type, it
        saves in the object instance the HTML attributes of the tag, where a 
        property of interest is contained (such as DATE of publication, AUTHORS,
        TEXT, TITLE of the article, etc.) 
        
        getAttributeDictionary: the respective getters of the above attribute
         
        downloadRawData: a method that downloads the whole HTML code from the
        specified URL. It may be part of the constructor method. It would make
        at least two things: send the get request and receive the result; extract
        the HTML code from the downloaded object.
        
        extractElement: a method that extracts a title/text/date/etc from a 
        "soup" given some attributes
        
        getElement: the getter for the thing that the above methods output
    '''
    @abstractmethod
    def __init__(self, url, auto_download = False):
        ''' The constructor method, obviously. The only argument it accepts is 
        the URL of the news article, which it saves as an object attribute.
        Auto_download = True if the user wants to download the HTML code 
        automatically
        '''
        pass

    @abstractmethod
    def downloadHTML(self):
        '''
        This method sends a request to the specified URL and "soupifies" the 
        result, saving it to an object attribute.
        '''
        pass
        
    @abstractmethod
    def getRawHTML(self):
        '''
        A Getter method; returns the raw ("soupified") HTML code.
        '''
        pass
    
    @abstractmethod
    def getExtractedElement(self):
        '''
        Self-explanatory; returns an already extracted element from the dictionary
        with all extracted elements. Raises a KeyError if the requested element 
        is not in the dictionary
        '''
        pass
        
    @abstractmethod
    def getExtractionPatterns(self):
        '''
        This method returns the extraction patterns used in self.extractAll()
        '''
        pass
    
    @abstractmethod
    def extractElement(self, tag_pattern):
        '''
        This method extracts the contents of the HTML tag specified by tag_pattern.
        
        An open question is whether it should save the result as a class attribute
        or return it like a Getter method.
        
        A consideration: if the element is saved as an object attribute then it
        would be easier to access it from WITHIN the Newspage class, as it would
        involve only looping over each of the Articles in the Newspage and in-
        voking the GETTER for the element of interest. 
        Otherwise, it would be the case that every time an element from an Ar-
        ticle is needed, the extraction procedure would have to be carried out.
        '''
        pass
    
    @abstractmethod
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
        '''
        pass
    
    @abstractmethod
    def __lt__(self, other):
        '''
        Defines the sorting rule for objects of the class Article. The natural
        rule is to sort by date, although the implementations may differ.
        '''
        pass




class Newspage(ABC):
    '''
    Also an Abstract Base Class for the Newspage class.
    
    A Newspage is a webpage which contains several news articles. The webpage
    is represented by its HTML code. The news articles' URL addresses are extrac-
    ted and saved in a separate object attribute. 
    
    It may also contain the HTML tag patterns that are required to extract ele-
    ments from an article. Therefore, the children of Newspage may have as their
    attributes the HTML tag patterns of the content of interest to the programmer.
    
    The main purpose of the Newspage class is to define an abstraction over the
    set of links-to-articles in a given webpage and the methods needed to extract
    data of interest from these links to articles.
    
    So far, the Newspage object will have the HTML tag patterns needed to:
    1. extract the URL-s of the articles on the Newspage page; 2. extract the
    elements of interest from the Newspage articles themselves. (they will be
    passed as arguments to the relevant methods of the Article family)
    
    Implicit is the assumption that the Newspage class will use the objects of
    the Article class and its children.
    
    Note: you may want to consider instantiating all class attributes in the 
    __init__() method with empty "whatevers". E.g. self.something = [], etc.
    '''
    
    @abstractmethod
    def __init__(self, url, auto_download = False):
        '''
        The class constructor, obviously. It isn't intended to do anything per 
        se, except for initializing the Newspage instance. Auto_download should
        be True if the user wants to download the HTML automatically.
        '''
        pass
    
    @abstractmethod
    def downloadHTML(self):
        '''
        This method downloads and "soupifies" the HTML code from the given URL.
        '''
        pass
    
    @abstractmethod
    def extractURLs(self):
        '''
        This method extract the article URL-s from the downloaded HTML code and
        stores them in an appropriate data structure.
        '''
        pass
    
    @abstractmethod
    def initializeArticles(self):
        '''
        This method uses the extracted by the above method URL-s and instantiates
        the respective Article objects.
        '''
        pass
    
    @abstractmethod
    def extractAllFromArticles(self):
        '''
        This method invokes Article.extractAll method for each article in self.articles.
        '''
        pass
    
    @abstractmethod
    def do_all(self):
        '''
        This method invokes the above four methods sequentially. It is solely for
        convenience, so it is arguable whether it should be defined as an abstract
        method.
        
        However, the Newspaper object should have an interface to work with a 
        Newspage and it might require a few lines more if this method here is not
        implemented.
        '''
        
    @abstractmethod
    def getRawHTML(self):
        '''
        A Getter method; returns the raw HTML code of the Newspage URL.
        Consider how to handle things if the downloadHTML() method hasn't been
        invoked before getRawHTML().
        '''
        pass
    
    @abstractmethod
    def getURLs(self):
        '''
        A Getter method; returns the extracted article URL-s.
        Again, consider handling of situations analogous to the above method
        '''
        pass
    
    @abstractmethod
    def getArticles(self):
        '''
        A Getter method; returns the Article objects that have been instantiated
        previously. Consider handling if initializeArticles() hasn't been invoked.
        '''
        pass
    
    @abstractmethod
    def getElementFromAllArticles(self):
        '''
        This method returns a specified element ('title', 'date', etc) from
        all articles
        '''
        pass
    


class Newspaper(ABC):
    ''' 
    This is the abstract base class for Newspapers.
    
    A Newspaper is a collection of Newspages, providing an abstraction over the
    Newspages' data (articles included) and provides an interface for working with
    these data.
    
    Formally, it should be sort of a collection of urls, each of which can be 
    interpreted as a Newspage, i.e. from it a Newspage object can be constructed and
    the latter's methods can be properly applied.
    
    Then, the class would initialize in itself various Newspages from the URL-s,
    which on their part would initialize the relevant Articles.
    
    Also, it should provide a high-level interface for working with Newspages 
    and Articles, i.e. downloading, extracting and getting the elements of interest
    from both Newspages and Articles.
    
    It is a matter of implementation for how exactly this collection of URL-s 
    would be generated. I.e. in the case of insidermedia.com the news pages for
    a given region follow a simple pattern that can be automated.
    On the other hand, if there are multiple topics, automatic generation of 
    URL-s may not be possible (i.e. {main_url}/sport, {main_url}/policics, etc.)
    '''
    
    @abstractmethod
    def __init__(self):
        '''
        Creates an object instance and instantiates all object attributes.
        '''
        pass
    
    @abstractmethod
    def generateURLs(self):
        '''
        Generates the URL-s of the individual Newspages as appropriate
        '''
        pass
    
    @abstractmethod    
    def generateNewspages(self):
        '''
        Creates a Newspage object from each URL. 
        
        It might also invoke the do_all() method of each Newspage, or calls 
        sequentially the do_all methods from Newspage - depends on whether do_all
        is implemented in all Newspages
        '''
        pass
        
    @abstractmethod
    def extractArticlesFromNewspages(self):
        '''
        This method is intended to call the initializeArticles and extractAll methods
        from Newspage. It assumes that Newspage.downloadHTML() and Newspage.extractURLs
        both have been called.
        '''
        pass
    
    @abstractmethod
    def getURLs(self):
        '''
        Returns all generated Newspage URL-s. 
        '''
        pass
        
    @abstractmethod
    def getNewspages(self):
        '''
        Returns all Newspages.
        '''
        pass
    
    @abstractmethod
    def getArticles(self):
        '''
        Returns all articles from all newspages
        '''
        pass
    
    @abstractmethod
    def getArticleElement(self):
        '''
        Returns a requested element from all articles in all newspages
        '''
        pass
        
#    @abstractmethod
#    def _makeURL(self):
#        pass







### https://medium.freecodecamp.org/how-to-scrape-websites-with-python-and-beautifulsoup-5946935d93fe

 

 
# 
#
#url = "https://www.insidermedia.com/insider/northwest/major-energy-project-to-support-north-west-job-creation"
#
##url = "http://www.bloomberg.com/quote/SPX:IND"
#
#url = 'https://www.insidermedia.com/insider/northwest/manchester-tech-incubator-launches'
#
#r = requests.get(url, verify = False)
#
# 
#
#raw = bs4.BeautifulSoup(r.text, 'html.parser')
#
# 
#
#title = raw.find('h1', {'itemprop':"headline"})
#
#print(title.text)
#
# 
#
#a_text = raw.find('div', {'class':'article-content', 'itemprop':'articleBody'})
#
#print(a_text)
#
# 
#
#for content in a_text.contents:
#
#    if type(content) == bs4.element.Tag:
#
#        if content.name == 'p':
#
#            print(content.text)
#
#           
#
#a_date = raw.find('time', {'itemprop':'datePublished'})
#
#print(a_date)

           

def extract_text_from_url(url):

    ''' This function downloads HTML content from an internet article and

    extracts the article title, text and date

    Parameters:

        url: string, a valid url address

        title_attributes: dictionary, contains the HTML attributes of the title

        text_attributes: dictionary, contains the HTML attributes of thea article body

        

    Returns:

        list, contains the title, text and date of the article

    '''

    ### Send the get request and collect the response

    response = requests.get(url, verify = False)

    ### Extract the HTML output

    raw_html = bs4.BeautifulSoup(response.text, 'html.parser')

    ### Extract the title

    article_title = raw.find('h1', {'itemprop':'headline'}).text

    ### Extract the text

    # Initialize the whole text string

    article_text = ''

    # Extract only the article body

    article_body = raw_html.find('div', {'class':'article-content', 'itemprop':'articleBody'})

    # Loop over the contents and add the text to the whole text string

    for content in article_body.contents:

        # Check if the child is a Tag element

        if type(content) == bs4.element.Tag:

            # If so, check if it is a new paragraph

            if content.name == 'p':

                # Add to whole text

                article_text += content.text

   

    ### Extract the date

    # Extract the date element of the HTML

    article_time = raw_html.find('time', {'itemprop':'datePublishe'})

    # Save the content as datetime

    date_published = datetime.datetime.strptime(article_time.text, '%d %b %Y')

    ### Create the output list

    output_list = [article_title, article_text, date_published]

    # Return

    return output_list

 

#print(extract_text_from_url(url))