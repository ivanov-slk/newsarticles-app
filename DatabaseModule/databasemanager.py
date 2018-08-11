# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:16:46 2018

@author: c14238a

"Cheat sheet":
    https://www.codementor.io/sheena/understanding-sqlalchemy-cheat-sheet-du107lawl
"""
#import os
#path = "C://Work//Python//Articles//DatabaseModule//"
#path = "D://Книги//Мои//Python//Articles//"
#os.chdir(path)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_
from sqlalchemy import Column, String, Integer, Date
from datetime import datetime
from DatabaseModule.article_orm import ArticleDatabaseModel
from DatabaseModule.base import engine, Session, Base

class DatabaseManager(object):
    '''
    This class manages the interactions with the article database. On initialization
    it establishes the connection and opens a session. It then manages the getting
    and storing data into the database.
    '''
    
    def __init__(self, engine, Session, Base):
        '''
        The constructor creates the engine, creates a session, sets the base, 
        initializes the base's metadata and, lastly, opens a session.
        '''
        # Create the engine
        self.engine = engine
        self.Base = Base
        # Create the metadata
        self.Base.metadata.create_all(self.engine)
        # Make a session
        self.session = Session()
        
    def getData(self, from_date, to_date):
        '''
        Gets data with the selected properties and returns it. Currently, it filters
        the entries only by date, but this functionality may need to be extended.
        
        Parameters:
            from_date: datetime.datetime object; the start date
            to_date: datetime.datetime object; the end date
        
        Returns:
            an iterable representing the entries in a DataFrame similar format
        '''
        ## The below code is not needed, as getData is not intended for user interaction
#        # If the dates are empty the whole period should be covered.
#        if from_date == '':
#            from_date = datetime.strptime('01 January 1970', '%d %B %Y')
#        else:
#            from_date = datetime.strftime(from_date, '%d %B %Y')
#        
#        if to_date == '':
#            to_date = datetime.now()
#        else:
#            to_date = datetime.strftime(to_date, '%d %B %Y')
        # Add 23:59 as the hour to to_date; otherwise it fails to pull the data
        # that has to_date as its date.
        from_date = from_date.replace(hour = 0, minute = 0, second = 0)
        to_date = to_date.replace(hour = 23, minute = 59, second = 59)
#        print(type(from_date))
#        print(type(to_date))
        # Make sure that the dates are stored in the database as datetimes, not strings or whatever        
#        articles_got = self.session.query(ArticleDatabaseModel).all()
        articles_got = self.session.query(ArticleDatabaseModel).filter(
                and_(ArticleDatabaseModel.date <= to_date, 
                     ArticleDatabaseModel.date >= from_date)).all()
        
        return articles_got
        
    def storeData(self, data_dictionary):
        '''
        Stores all downloaded data into the database. The code should be extended
        so as to store only the non-duplicated articles. Consider constructing
        a unique ID for each article that is going to be compared on saving.
        '''            
        # Create a list with the database entries and fill it
        articles_list = []
        
        for i in range(len(data_dictionary['title'])):    
            article_obj = ArticleDatabaseModel(title = data_dictionary['title'][i], 
                                  text = data_dictionary['text'][i], 
                                  topic = data_dictionary['topic'][i], 
                                  region = data_dictionary['region'][i], 
                                  date = data_dictionary['date'][i], 
                                  url = data_dictionary['url'][i],
                                  keep_flag = data_dictionary['keep_flag'][i]
                                  )
            articles_list.append(article_obj)    
        
        # Check if a downloaded article is already in the database (titles serve as
        # id here) and remove anu such article from the list
        # Yeah, I know that this stuff is O(n^2)... tell me if there is a better solution, 
        # I'm lazy right now.
        current_records = self.session.query(ArticleDatabaseModel).all()
        
        loop_list = articles_list[:]
        for article in loop_list:
            for record in current_records:
                if article.title == record.title:
                # meaning that this article is already stored in the database
#                    print(article.title)
#                    print(record.title)
#                    print(article.keep_flag)
#                    print(record.keep_flag)
                    if article.keep_flag != record.keep_flag:
                        # Try updating the existing records with the new keep_flag
                        # and then removing the existing record from the articles_list
                        # that is going to be saved into the database later.
                        self.session.query(ArticleDatabaseModel).filter(
                                ArticleDatabaseModel.title == article.title).update(
                                        {'keep_flag': article.keep_flag})
                        self.session.commit()
#                        print('here')
#                        print()
#                        print()
                    # in any case remove the article from the article list - it already 
                    # exists in the database no matter whether its keep_flag has been
                    # updated or not.
                    articles_list.remove(article)
                    
        if articles_list == []:
            return
        
        # Add to session and commit    
        for article in articles_list:
            self.session.add(article)
        
        self.session.commit()
#        self.session.close()
    

database_manager = DatabaseManager(engine = engine, Session = Session, Base = Base)
#    
#articles_list = []
#
#for i in range(len(final_df)):
#    article_obj = ArticleDatabaseModel(title = final_df.title[i], 
#                          text = final_df.text[i], 
#                          topic = final_df.topic[i], 
#                          region = final_df.location[i], 
#                          date_published = final_df.date[i], 
#                          url = final_df.url[i]
#                          )
#    articles_list.append(article_obj)    
#    
#    
#for article in articles_list:
#    session.add(article)
#    
#session.commit()
#session.close()
#
#arts = session.query(ArticleDatabaseModel).all()
#for art in arts:
#    print(art.title, art.url, art.topic, art.id)