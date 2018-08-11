# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:08:25 2018

@author: c14238a
"""

#import os
#path = "C://Work//Python//Articles//DatabaseModule//"
##path = "D://Книги//Мои//Python//Articles//"
#os.chdir(path)


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('sqlite:///articles.db')
Session = sessionmaker(bind=engine)

Base = declarative_base()