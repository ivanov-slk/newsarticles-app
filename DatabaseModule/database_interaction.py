# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:16:46 2018

@author: c14238a
"""
import os
path = "C://Work//Python//Articles//DatabaseModule//"
#path = "D://Книги//Мои//Python//Articles//"
os.chdir(path)

from datetime import date
from article_orm import Article_db
from base import Session, engine, Base

Base.metadata.create_all(engine)

session = Session()

articles_list = []

for i in range(len(final_df)):
    article_obj = Article_db(title = final_df.title[i], 
                          text = final_df.text[i], 
                          topic = final_df.topic[i], 
                          region = final_df.location[i], 
                          date_published = final_df.date[i], 
                          url = final_df.url[i]
                          )
    articles_list.append(article_obj)    
    
    
for article in articles_list:
    session.add(article)
    
session.commit()
session.close()

arts = session.query(Article_db).all()
for art in arts:
    print(art.title, art.url, art.topic, art.id)