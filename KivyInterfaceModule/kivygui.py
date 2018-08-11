'''
Code used from the official example on kivy.ogr
RecycleView - full implementation:
    https://kivy.org/docs/_modules/kivy/uix/recycleview.html
    
    
To-Do list:

Keep/Discard buttons don't work AS EXPECTED when an item is not visible on the 
list. This shouldn't be a problem, since the user has to (well, it is good to 
and he is supposed to) scroll down/up to actually see the title before clicking 
Keep/Discard.

Consider adding a "Caveats" field like the Text label

    

Caveats:
    Mind that there are links to the so called InsiderNewspage in 
the GeneralNewspaper (the InsiderNewspage should be general enough, but still...).
This is fine for now, but if other Newspapers are addeds some refactoring would 
be needed. The point is that GeneralNewspaper is not "general" right now, as it 
refers to InsiderNewspage, so that reference might be replaced with an instance 
variable representing the specific Newspage needed.

    Some modules may be imported more than once. This shouldn't be a problem, but
causes some inefficiencies. You should keep this in mind anyway.

    Changed the article ORM date to Date format.
    Also added keep_flag to Newsarticle; Article_ORM and DatabaseManager
'''
# Change the current working directory
#import os
#os.chdir("D:/Книги//Мои//Python//Articles//Articles//")
#os.chdir("C://Work//Python//Articles//")
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath) + '\\'
#os.chdir(dname)

from KivyInterfaceModule.widgets import *
from KivyInterfaceModule.linker import *
from datetime import datetime

'''
1. the left pane should contain text inputs for the from-date and to-date;
        a button that downloads the data from the internet; a button that gets
        the data from the database and a button that stores the downloaded data
        into the database.
'''


class KivyGUIApp(App, ModuleLinker):
    '''
    This is the main application.
    
    It inherits from `App` and `ModuleLinker`. 
    This is needed, as the KivyGUIApp class integrates the graphical user inter-
    face and the actual application functionalities; thus requires some sort of 
    communication between the GUI and the other modules. This communication is 
    carried out by the ModuleLinker class, which provides the necessary properties
    and methods to do that.
    '''
    
#    data = [{'text': str(x)} for x in range(100)]
#    title = "random"
    
    def __init__(self, **kwargs):
        super(KivyGUIApp, self).__init__(**kwargs)
        self.text = kwargs['text']
        self.title = kwargs['title']
        self.message_box_factory = MessageBoxFactory()
        self.selected_title = ''
        self.selected_index = None
        
    def build(self):
        # Empty list for the RecycleView - it will not be clickable, which is the point.
        self.data = []
        # Initialize the widgets
        self.main_window = MainWindow()
        self.main_window.add_widget(LeftPane())
        self.main_window.add_widget(CentralPane(data = self.data)) #[{'text':element} for element in 
#                                 self.extracted_elements_dictionary['title']]))
        self.main_window.add_widget(RightPane())
        return self.main_window
    
    
#    title, message = "title download", "message download"
    def createMessageBox(self, title, message, the_type = 'default'):
        '''
        This method handles the pop-up for all panes.
        
        Parameters:
            title: string; the title for the message box
            message: string; the message of the message box
        '''
        self.message_box = self.message_box_factory.getMessageBox(the_type = the_type)
        self.message_box.title = title
        self.message_box.message = message
        self.message_box.open() 
        print('test press: ', title, message)


    def downloadDataButton(self):
        '''
        Downloads data from the internet within the specified time period.
        '''
        from_date = self.main_window.children[2].ids.from_dt.text
        to_date = self.main_window.children[2].ids.to_dt.text
        
        # If the dates are empty the whole period should be covered.
        if from_date == '':
            from_date = datetime.strptime('01 January 1970', '%d %B %Y')
        else:
            try:
                from_date = datetime.strptime(from_date, '%d %B %Y')
            except ValueError:
                self.createMessageBox(title = 'Wrong date format!', 
                             message = 'Please enter a date in format dd mmmm yyyy')
                return
        
        if to_date == '':
            to_date = datetime.now()
        else:
            try:
                to_date = datetime.strptime(to_date, '%d %B %Y')
            except ValueError:
                self.createMessageBox(title = 'Wrong date format!', 
                             message = 'Please enter a date in format dd mmmm yyyy')
                return
        
        # Check if from_date is before to_date
        if from_date > to_date:
            self.createMessageBox(title = 'Wrong date format!',
                             message = "End date can't be before start date!")
        # Invoke the download data method
        self.downloadData(from_date, to_date)
        # Get and store the new article element(s)
        self.getElementsDictionary(source = 'downloaded')
        # Update self.data
        self.main_window.children[1].data = [{'text':element} for element in 
                                 self.extracted_elements_dictionary['title']]
        self.createMessageBox('Done!', 'Download successful!')
        
    
    def pullFromDatabaseButton(self):
        '''
        A wrapper for the respective ModuleLinker method. It checks the format
        of the dates and shows a message box if it is wrong.
        '''
        from_date = self.main_window.children[2].ids.from_dt.text
        to_date = self.main_window.children[2].ids.to_dt.text
        
        # If the dates are empty the whole period should be covered.
        if from_date == '':
            from_date = datetime.strptime('01 January 1970', '%d %B %Y')
        else:
            try:
                from_date = datetime.strptime(from_date, '%d %B %Y')
            except ValueError:
                self.createMessageBox(title = 'Wrong date format!', 
                             message = 'Please enter a date in format dd mmmm yyyy')
                return
        
        if to_date == '':
            to_date = datetime.now()
        else:
            try:
                to_date = datetime.strptime(to_date, '%d %B %Y')
            except ValueError:
                self.createMessageBox(title = 'Wrong date format!', 
                             message = 'Please enter a date in format dd mmmm yyyy')
                return
        
        # Get the articles from the database
        self.pullFromDatabase(from_date = from_date, to_date = to_date) 
        # Check if any data was pulled
        if self.extracted_elements_dictionary == {}:
            self.createMessageBox('Oops!', 'No data has been pulled from the database!')
            return
        # Update self.data
        self.main_window.children[1].data = [{'text':element} for element in 
                                 self.extracted_elements_dictionary['title']]
        
        self.createMessageBox('Done!', 'Pull successful!')

    def storeDataButton(self):
        '''
        This stores the downloaded data to the database.
        '''
        self.saveToDatabase()
        self.createMessageBox('Done!', 'Save successful!')
        
    def keepButton(self):
        '''
        Handles what happens when the 'Keep' button is pressed - updates the
        keep_flag element of the elements dictionary with a 1
        '''
        try:
            article_index = self.extracted_elements_dictionary['title'].index(self.selected_title)
        except ValueError:
            if self.selected_title == '':
                self.createMessageBox(title = 'Oops!', message = 'Please select something.')
                return
            else:
                self.createMessageBox(title = 'Oops!',
                                 message = 'No such element in the list: ' + str(self.selected_title))
            return
        
        self.setArticleFlag(article_index = article_index, flag = 1)
        # Checks
#        print(self.selected_title)
#        print(article_index)
#        print(self.extracted_elements_dictionary['title'])
#        print(self.extracted_elements_dictionary['keep_flag'])
        
#        # Select next item; moved to the .kv
#        self._selectNextItem()


    def discardButton(self):
        '''
        Handles what happens when the 'Discard' button is pressed - updates the
        keep_flag element of the elements dictionary with a 0
        '''
        # Check if something is selected
        try:
            article_index = self.extracted_elements_dictionary['title'].index(self.selected_title)
        except ValueError:
            if self.selected_title == '':
                self.createMessageBox(title = 'Oops!', message = 'Please select something.')
                return
            else:
                self.createMessageBox(title = 'Oops!',
                                 message = 'No such element in the list: ' + str(self.selected_title))
            return
        
        self.setArticleFlag(article_index = article_index, flag = 0)
        # Checks
#        print(self.selected_title)
#        print(article_index)
#        print(self.extracted_elements_dictionary['title'])
#        print(self.extracted_elements_dictionary['keep_flag'])
        
#        # Select next item; moved to the .kv 
#        self._selectNextItem()
        
        
    def detailsButton(self):
        '''
        Handles what happens when the 'Details' button is pressed - 
        '''
        self.createMessageBox(title = 'Method not yet implemented... ', 
                         message = "... and perhaps won't be... ")
        
    def _updateDetails(self):
        '''
        This should be a private method that updates the article details on 
        selection change.
        '''
        # Set the right pane as an object for convenience
        right_pane = self.main_window.children[0]
        
        # Update the region
        right_pane.ids.region_label.text = "Region: \n\n" + \
                self.extracted_elements_dictionary['region'][self.selected_index]
        
        # Update the original topic
        right_pane.ids.topic_label.text = "Topic: \n\n" + \
                self.extracted_elements_dictionary['topic'][self.selected_index]
                
        # Update the date
        right_pane.ids.date_label.text = "Date published: \n\n" + \
datetime.strftime(self.extracted_elements_dictionary['date'][self.selected_index], '%d %B %Y')
                
        # Update the polarity score
        right_pane.ids.polarity_label.text = "Polarity score: \n\n" + str(self.nlpprocessor \
                                             .getPolarityScore(self.selected_index))

        # Update the article text
        right_pane.ids.text_label.text = "Text: \n\n" + \
                self.extracted_elements_dictionary['text'][self.selected_index]
                
        # Update the emotions table
        right_pane.ids.emotions_label.text = "Emotions, %: \n\n" + \
                        str(self.nlpprocessor.getSentiments(self.selected_index))
        
        # Update the word frequencies
        right_pane.ids.frequencies_label.text = "Word frequencies, %: \n\n" + \
                        str(self.nlpprocessor.getFrequencies(self.selected_index, 10))
        
        # Update the LDA current model keywords
        right_pane.ids.lda_keywords_label.text = "Keywords (LDA, current): \n\n" + \
                        '\n'.join(self.nlpprocessor.getTopicLDA(self.selected_index))
        
        # Update the LDA database model keywords
        right_pane.ids.lda_keywords_db_label.text = "Keywords (LDA, database): \n\n" + \
                        '\n'.join(self.nlpprocessor_db.getTopicLDA(self.selected_index))
        
        # Update the NMF current model keywords
        right_pane.ids.nmf_keywords_label.text = "Keywords (NMF, current): \n\n" + \
                        '\n'.join(self.nlpprocessor.getTopicNMF(self.selected_index))
        
        # UPdate the NMF database model keywords
        right_pane.ids.nmf_keywords_db_label.text = "Keywords (NMF, database): \n\n" + \
                        '\n'.join(self.nlpprocessor_db.getTopicNMF(self.selected_index))
        
        # Update the NER elements
        right_pane.ids.ner_gpe_label.text = "Geo-political entities: \n\n" + \
                        '\n'.join(self.nlpprocessor.getNamedEntities(self.selected_index, 'GPE'))
        right_pane.ids.ner_org_label.text = "Organizations: \n\n" + \
                        '\n'.join(self.nlpprocessor.getNamedEntities(self.selected_index, 'Organization'))
        right_pane.ids.ner_person_label.text = "Persons: \n\n" + \
                        '\n'.join(self.nlpprocessor.getNamedEntities(self.selected_index, 'Person'))
        right_pane.ids.ner_loc_label.text = "Locations: \n\n" + \
                        '\n'.join(self.nlpprocessor.getNamedEntities(self.selected_index, 'Location'))
        right_pane.ids.ner_other_label.text = "Others: \n\n" + \
                        '\n'.join(self.nlpprocessor.getNamedEntities(self.selected_index, 'Other'))
        
        
        # Update the flag
        # Convert the flag into string
        if self.extracted_elements_dictionary['keep_flag'][self.selected_index] == 1:
            current_flag = 'Keep'
        elif self.extracted_elements_dictionary['keep_flag'][self.selected_index] == 0:
            current_flag = 'Discard'
        else:
            raise ValueError('A keep_flag value seems to be incorrect.')
        right_pane.ids.keep_flag_label.text = "Current flag: \n\n" + current_flag

    def _selectNextItem(self):
        '''
        This selects the next item in the RecycleView list (or whatever it is called)
        externally, e.g. from an "outside" button
        '''
        next_index = None # the next index value that is going to be determined below
        # Get the length of the list of items, assuming that all lists of the
        # elements dictionary are of equal length
        data_length = len(self.extracted_elements_dictionary['title'])
        # Check which is the selected item and handle appropriately
        if self.selected_index == data_length - 1: # indices start at 0
            next_index = 0
        elif self.selected_index == None:
            next_index = 0
        else: 
            next_index = self.selected_index + 1
        # Select the node
#        print()
#        print()
#        print(data_length)
#        print(next_index)
#        print()
#        print()
        self.selected_index = next_index
        self.main_window.children[1].layout_manager.select_node(next_index)


### Tests.
#text = 'empty '
#title = 'Articles'
##data = [{'text': str(x)} for x in range(100)]
#kvapp = KivyGUIApp(text=text, title=title)
#kvapp.run()

'''
Completed / Redundant items from the To-Do list

Say, put in self.data ALL article data in a dictionary (i.e. all elements from 
all articles). On Keep / Discard find the index of the title in the list of titles
and change the value of the corresponding index in the flags list.
Thus, Keep / Discard will take the title as argument, as well as the value of 
the flag, and search in self.data for the corresponding flag to update it.

For this, you should implement a method that pulls all data into a dictionary
from both article_orm objects and newspaper objects and then puts it into self.data.
It would be either in self.database_articles or in self.newspapers.
Also, think of a method that receives variable number of keyword arguments that
will reference to an article element (i.e. it should be flexible if some article
elements are removed or added).

If the user clicks on an article detail a pop-up should appear with the full detail.                                      

The details button may be kind of redundant... it is more convenient to build
a separate boxlayout on the main screen where all details will be updated on 
RecycleView selection change.


Check the flush error - it won't overwrite the database record because of conflicting
primary keys, but I want to overwrite! - done

Consider automatically moving to the next item in the list when Keep/Discard
is clicked. - done

Finished the modelling module (the LDA and NMF don't produce the best results, but...)

Next thing in the list: consider where to instantiate the NLP processor and how
to invoke its methods. Then update the labels in the Right Pane.
'''