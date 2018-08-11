import os
os.chdir("D:/Книги//Мои//Python//Articles//Articles//KivyGUI")


import kivy

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

'''
1. the left pane should contain text inputs for the from-date and to-date;
        a button that downloads the data from the internet; a button that gets
        the data from the database and a button that stores the downloaded data
        into the database.
'''


class MainWindow(BoxLayout):
    '''
    This defines the main window. The main window should have three parts: 
        1. a left pane with buttons and text inputs;
        2. a central pane where certain text output should be displayed;
        3. a right pane with buttons for working with the central pane
        
    Uses horizontal box layout to contain the three panes
    '''
    pass

class Box1(BoxLayout):
    pass

class Box2(BoxLayout):
    pass

class KivyGUIApp(App):
    '''
    This is the main application.
    '''
    
    def build(self):
        mw = MainWindow()
        mw.add_widget(Button(text = "adding..."))
        mw.add_widget(BoxLayout(orientation = "vertical"))
        mw.add_widget(Box1(orientation = "vertical"))
        mw.add_widget(Box2(orientation = "vertical"))
        return mw
    
kvapp = KivyGUIApp()
kvapp.run()