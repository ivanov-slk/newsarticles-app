# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:03:39 2018

@author: c14238a
"""

from sqlalchemy import *
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
 
engine = create_engine('sqlite:///student.db', echo=True)
Base = declarative_base()
 
########################################################################
class Student(Base):
    """"""
    __tablename__ = "student"
 
    id = Column(Integer, primary_key=True)
    username = Column(String)
    firstname = Column(String)
    lastname = Column(String)
    university = Column(String)
 
    #----------------------------------------------------------------------
    def __init__(self, username, firstname, lastname, university):
        """"""
        self.username = username
        self.firstname = firstname
        self.lastname = lastname
        self.university = university
 
# create tables
Base.metadata.create_all(engine)




import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from tabledef import *
 
engine = create_engine('sqlite:///student.db', echo=True)
 
# create a Session
Session = sessionmaker(bind=engine)
session = Session()
 
# Create objects  
user = Student("james","James","Boogie","MIT")
session.add(user)
 
user = Student("lara","Lara","Miami","UU")
session.add(user)
 
user = Student("eric","Eric","York","Stanford")
session.add(user)
 
# commit the record the database
session.commit()



import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from tabledef import *
 
engine = create_engine('sqlite:///student.db', echo=True)
 
# create a Session
Session = sessionmaker(bind=engine)
session = Session()
 
# Create objects  
for student in session.query(Student).order_by(Student.id):
    print(student.firstname, student.lastname)
    
    
    
    



import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
#from tabledef import *
 
engine = create_engine('sqlite:///student.db', echo=True)
 
# create a Session
Session = sessionmaker(bind=engine)
session = Session()
 
# Select objects  
for student in session.query(Student).filter(Student.firstname == 'Eric'):
    print(student.firstname, student.lastname)