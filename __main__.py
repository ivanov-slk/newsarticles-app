# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:39:28 2018

@author: C14238A
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath) + '\\'
os.chdir(dname)

from KivyInterfaceModule.kivygui import *

text = 'nothing found'
title = 'Articles'
kivy_app = KivyGUIApp(text=text, title=title)
kivy_app.run()