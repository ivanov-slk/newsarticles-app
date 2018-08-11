# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:10:35 2018

@author: c14238a
"""

from sqlalchemy import Column, String, Integer, DateTime
from DatabaseModule.base import Base

class ArticleDatabaseModel(Base):
    ''' 
    This class is intended to map the article data to the database.
    '''
    
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, primary_key = False)
    text = Column(String)
    topic = Column(String)
    region = Column(String)
    date = Column(DateTime)
    url = Column(String)
    keep_flag = Column(Integer)
    
    def __init__(self, title, text, topic, region, date, url, keep_flag):
        self.title = title
        self.text = text
        self.topic = topic
        self.region = region
        self.date = date
        self.url = url
        self.keep_flag = keep_flag
    
    