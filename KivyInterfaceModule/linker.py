#import os
#os.chdir("D:/Книги//Мои//Python//Articles//Articles//")
#os.chdir("C://Work//Python//Articles//")

from CollectionModule.SpecificNewspapers.insidernewspaper import *
from DatabaseModule.databasemanager import database_manager
from ModellingModule.implemented import InsiderNLP
from datetime import datetime

class ModuleLinker(object): 
    '''
    This class ensures the proper communication between the graphical user inter-
    face and the other modules. It takes care of the initialization of the article
    collection, modelling and database modules, as well as their proper workflow.
    
    A note about the NLP processor. I'm wondering how exactly to instantiate it. 
    It fails if I instantiate it in __init__() as it doesn't have any text to store
    in itself. I can instantiate it with an empty list of documents, but this may 
    break if method calls are done (although the user is not supposed to use this
    in any way from the GUI). Then I can write a Setter for the articles... which
    may be a good idea if the user works with different set of articles within 
    a single "session" of the application. E.g. if he downloads multiple times
    different articles. 
    
    Wait! I actually need two separate models available at the same time - the 
    "current" and the "database" model. So I'll have to instantiate two NLP processor
    objects only to use the two different models I need... not the best solution, eh?
    
    Currently, the ModuleLinker will instantiate two NLP processor objects on 
    __init__() with empty raw data. The raw data will be set on downloading or on 
    pulling from database.
    '''
    
    def __init__(self, **kwargs):
        '''
        Initializes the three modules = Collection, Modelling, Database. 
        
        The collection is represented by a InsiderNewspaper instances for every
        region in insidermedia's nomenclature.
        '''
        # tests
#        self.data = [{'text':'nothing here' }, {'text':'another nothing here'}]
#        self.title = kwargs['title']
        ### Initialize the collection module objects
        self.newspapers = []
        
        regions = ['central-and-east', 'ireland', 'midlands', 'northeast', 'northwest', 
          'southeast', 'southwest', 'wales', 'yorkshire']
        regions = ['southeast']
        for region in regions:
            self.newspapers.append(InsiderNewspaper(region = region))
            
        self.extracted_elements_dictionary = {}
        
        ### Initialize the modelling module objects
        self.nlpprocessor = InsiderNLP([])
        self.nlpprocessor_db = InsiderNLP([])
        
        ### Initialize the database module objects
        self.database_manager = database_manager
        self.database_articles = []
        
    def downloadData(self, from_date, to_date):
        '''
        Downloads and extracts articles from the newspaper(s). It just invokes
        the appropriate Newspaper methods.
        '''
        for newspaper in self.newspapers:
            newspaper.setFromDate(from_date)
            newspaper.setToDate(to_date)
            newspaper.generateNewspages()
            newspaper.extractArticlesFromNewspages()
            
        # Store the data
        self.getElementsDictionary(source = 'downloaded')
        self._modelTheData()
        
    def pullFromDatabase(self, from_date, to_date):
        '''
        Gets from the database the articles that fit within the specified start 
        and end date. If from_date is left blank it is assumed to be 1 January 1970; 
        if to_date is left blank it is assumed to be today's datetime.
        
        Parameters:
            from_date: datetime.datetime object; the start date
            to_date: datetime.datetime object; the end date
            
        Returns:
            saves to self the database table
        '''
        ## The below code is not needed, as getData is not intended for user interaction
#        # If the dates are empty the whole period should be covered.
#        if from_date == '':
#            from_date = datetime.strptime('01 January 1970', '%d %B %Y')
#        else:
#            from_date = datetime.strptime(from_date, '%d %B %Y')
#        
#        if to_date == '':
#            to_date = datetime.now()
#        else:
#            to_date = datetime.strptime(to_date, '%d %B %Y')
            
        # Get the data
        self.database_articles = self.database_manager.getData(from_date, to_date)
#        print(type(from_date))
#        print(type(to_date))
        # Store the data
        self.getElementsDictionary()
        self._modelTheData()
        
    def saveToDatabase(self):
        '''
        Saves all downloaded articles to the database. They should be in
        self.extracted_elements_dictionary
        '''
        self.database_manager.storeData(self.extracted_elements_dictionary)
        
    def _modelTheData(self):
        '''
        Private method. Calls the relevant methods in the NLP processor that 
        produce the model results.
        Do I really need a separate method that calls another method of an object
        that is already stored in self?
        
        Note that in __init__() no data is downloaded nor extracted - there 
        is nothing to model yet. So the modelling will be done separately. For
        example, this method may be invoked at the end of the downloadData() method
        and the pullFromDatabase() method.
        
        '''
        self.nlpprocessor.setRawData(self.extracted_elements_dictionary['text'])
        self.nlpprocessor.doAll()
        self.nlpprocessor_db.setRawData(self.extracted_elements_dictionary['text'])
        self.nlpprocessor_db.clean()
        self.nlpprocessor_db.predictTopicLDA()
        self.nlpprocessor_db.predictTopicNMF()
    
    def getPulledArticleElement(self, element_name):
        '''
        This method requires that some data has been pulled from the database.
        As self.pullDataFromDatabase returns a list of ArticleDatabaseModel objects, 
        they have to be put into a reasonable format. This method gets a specified
        element and returns the field that corresponds to it as a list.
        
        The corresponding button on the GUI first calls the method that pulls
        data from the database, so the first sentence of the docstring shouldn't
        cause any issues... 
        
        A note: getElementDictionary notifies if no articles are pulled from the
        database, but doesn't notify if nothing has been downloaded.
        
        Parameters:
            element_name: string, valid element name
            
        Returns:
            a list with the respective element from all articles
        '''
        return_list = []
        
        for article_model in self.database_articles:
            # getattr gets object attribute from string
            # append as article_model.whatever is a single element
            return_list.append(getattr(article_model, element_name))
            
        return return_list
    
    def getDownloadedArticleElement(self, element_name):
        '''
        This method is analogous to the above one, except for this one gets elements
        from a downloaded article. It requires that some data has been already 
        downloaded.
        
        Parameters:
            element_name: string, valid element name
            
        Returns:
            a list with the respective element from all articles
        '''
        return_list = []
        
        for newspaper in self.newspapers:
            # extend as newspaper.getArticleelement() returns a list and we 
            # don't want nested lists
            return_list.extend(newspaper.getArticleElement(element_name))
            
        return return_list
        
    def getElementsDictionary(self, source = 'database', return_dictionary = False):
        '''
        This method returns a dictionary of all article elements that are contained
        either in self.newspapers or self.database_articles. It is basically a
        wrapper of getPulledArticleElement and getDownloadedArticleElement.
        
        Parameters:
            source: string; either 'database' or 'downloaded'
            return_dictionary: boolean; True to return a dictionary, False to 
            store the dictionary to an instance variable
            
        Returns:
            dictionary; contains key-value pairs of 'element_name':list_of_values
        '''
        # The elements to be extracted
        extracted_elements_names = ['title', 'text', 'topic', 'region', 'date', 'url', 'keep_flag']        
        extracted_elements_dictionary = {'title':['nothing found'],
                                         'text':['nothing found'],
                                         'topic':['nothing found'],
                                         'region':['nothing found'],
                                         'date':[datetime.now()],
                                         'url':['nothing found'],
                                         'keep_flag':[0]}
        
        # Check[ what source has been requested and call the appropriate method
        if source == 'database':
            # Notify if the source is empty and return
            if self.database_articles == []:
                print('Oops! No data has been pulled from the database!')
                if return_dictionary == False:
                    self.extracted_elements_dictionary = extracted_elements_dictionary
                return
            for element in extracted_elements_names:
                extracted_elements_dictionary[element] = self.getPulledArticleElement(element)
                
        elif source == 'downloaded':
            for element in extracted_elements_names:
                extracted_elements_dictionary[element] = self.getDownloadedArticleElement(element)
            
        else:
            raise ValueError('Unknown source requested. Please choose only "database" or "downloaded".')
        
        if return_dictionary == True:
            return extracted_elements_dictionary
        else:
            self.extracted_elements_dictionary = extracted_elements_dictionary
       
    def setArticleFlag(self, article_index, flag):
        '''
        This method sets the flag of an article. The flag value should be either
        0 or 1, where 0 means 'discard' and 1 means 'keep'.
        
        Parameters:
            article_index: integer; the index of the article element in the list 
            of all article elements of a given type (say, the index of the title
            of an article in the list of all article titles)
            
            flag: integer; 0 or 1. 
        
        Returns:
            nothing; changes an article's flag
        '''
        if flag != 0 and flag != 1:
            raise ValueError('The value of the flag should be either 0 or 1')
            
        self.extracted_elements_dictionary['keep_flag'][article_index] = flag
        
        
    # Test methods
    def getA(self):
        return self.data
    def setA(self, a):
        self.data = a
    def getTitle(self):
        return self.title
    def setTitle(self, title):
        self.title = title
    def testLinkages(self):
        print('I was called from elsewhere!')
        
        
        
#### Tests
#title = 'random title'
#from_date = '29 June 2018'
#to_date = '29 June 2018'
#
#lnk = ModuleLinker(title = title, database_manager = database_manager)
#lnk.downloadData(from_date = from_date, to_date = to_date)
#lnk.saveToDatabase()
#lnk.pullFromDatabase()