import kivy

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
from kivy.uix.recycleview import RecycleView
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.button import Button
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.properties import ListProperty, StringProperty, ObjectProperty, BooleanProperty
from kivy.core.window import Window
from kivy.clock import Clock

class MessageBoxFactory(object):
    '''
    This should be sort of a MessageBox factory. It takes the type of message
    box needed and returns a MessageBox(Popup) object.
    
    Currently supported types:
        'default': a default message box with title, message and button OK
        'ok_cancel': message box with title and message and two buttons (OK/Cancel)
    '''
    def __init__(self):
        super(MessageBoxFactory, self).__init__()
    
    def getMessageBox(self, the_type):
        if the_type == 'default':
            return MessageBox()
        elif the_type == 'ok/cancel':
            return MessageBoxOKCancel()
        elif the_type == 'textbox':
            return MessageBoxText()
        
class MessageBox(Popup):
    '''
    The standard message box - title; message; OK button that closes on click.
    '''
    title = StringProperty()
    message = StringProperty()
    
class MessageBoxOKCancel(Popup):
    '''
    This message box has two buttons - OK and Cancel. OK allows the program to 
    continue what it has been doing prior to the message box popping up; cancel
    stops the program and returns to the GUI.
    
    The class is not ready for use.
    '''
    title = StringProperty()
    message = StringProperty()
    
    def ok_clicked(self):
        self.dismiss()
        
    
    def cancel_clicked(self):
        self.dismiss()
        return False

class MessageBoxText(Popup):
    '''
    This "message box" provides only a text output without any buttons.
    '''
    text = StringProperty()
    title = StringProperty()
    message = StringProperty()
    

class HoverBehavior(object):
    """Hover behavior.
    :Events:
        `on_enter`
            Fired when mouse enter the bbox of the widget.
        `on_leave`
            Fired when the mouse exit the widget 
    """

    hovered = BooleanProperty(False)
    border_point= ObjectProperty(None)
    '''Contains the last relevant point received by the Hoverable. This can
    be used in `on_enter` or `on_leave` in order to know where was dispatched the event.
    '''

    def __init__(self, **kwargs):
        self.register_event_type('on_enter')
        self.register_event_type('on_leave')
        Window.bind(mouse_pos=self.on_mouse_pos)
        super(HoverBehavior, self).__init__(**kwargs)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return # do proceed if I'm not displayed <=> If have no parent
        pos = args[1]
        #Next line to_widget allow to compensate for relative layout
        inside = self.collide_point(*self.to_widget(*pos))
        if self.hovered == inside:
            #We have already done what was needed
            return
        self.border_point = pos
        self.hovered = inside
        if inside:
            self.dispatch('on_enter')
        else:
            self.dispatch('on_leave')

    def on_enter(self):
        pass

    def on_leave(self):
        pass

from kivy.factory import Factory
Factory.register('HoverBehavior', HoverBehavior)

#class Tooltip(Label):
#    pass
#
class CustomLabel(Label, HoverBehavior):
    '''
    An ordinary label, just the on_touch_down method has been modified so as to
    produce a message box only when the region of the widget is clicked.
    '''
    def on_enter(self, *args):
        print("You are in, through this point: ", self.border_point)
        app = App.get_running_app()
        # Check if something is selected and exit if nothing is.
        if app.selected_index == None:
            return
        # Get the title and the message of the message box
        title = app.extracted_elements_dictionary['title'][app.selected_index]
        message = app.extracted_elements_dictionary['text'][app.selected_index]
        # Call the message box creating method
        app.createMessageBox(title = title, message = message, the_type = 'textbox')

    def on_leave(self, *args):
        print("You left through this point: ", self.border_point)
        app = App.get_running_app()
        # Check if something is selected and exit if nothing is.
        if app.selected_index == None:
            return
        app.message_box.dismiss()


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    '''
    This should create a box layout to the recycle box and add layout selection
    behaviour options...?
    
    source:
    https://stackoverflow.com/questions/43836472/select-next-previous-kivy-recycleview-row-on-button-press
    '''
#    def get_nodes(self):
#        nodes = self.get_selectable_nodes()
#        if self.nodes_order_reversed:
#            nodes = nodes[::-1]
#        if not nodes:
#            return None, None
#
#        selected = self.selected_nodes
#        if not selected:  # nothing selected, select the first
#            self.select_node(nodes[0])
#            return None, None
#
#        if len(nodes) == 1:  # the only selectable node is selected already
#            return None, None
#
#        last = nodes.index(selected[-1])
#        self.clear_selection()
#        return last, nodes
#
#    def select_next(self):
#        last, nodes = self.get_nodes()
#        if not nodes:
#            return
#
#        if last == len(nodes) - 1:
#            self.select_node(nodes[0])
#        else:
#            self.select_node(nodes[last + 1])
#
#    def select_previous(self):
#        last, nodes = self.get_nodes()
#        if not nodes:
#            return
#
#        if not last:
#            self.select_node(nodes[-1])
#        else:
#            self.select_node(nodes[last - 1])
    

class SelectableLabel(RecycleDataViewBehavior, Label):
    '''
    This should implement a label that can be selected (via the  Recycle... parent)
    in the view. 
    '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    
    def refresh_view_attrs(self, rv, index, data):
        '''
        This method is called when the data dictionary has changed.
        '''
        self.index = index
        print()
        print('Refreshing screen perhaps?')
        print()
        return super(SelectableLabel, self).refresh_view_attrs(rv, index, data)
    
    def on_touch_down(self, touch):
        '''
        Handles what happens (with the Label?) when a label is pressed
        '''
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)
        
    def apply_selection(self, rv, index, is_selected):
        '''
        Respond to the selection of items in the view... somehow. And also store
        the currently selected item (title) in the app's attributes.
        '''
        self.selected = is_selected
        if is_selected:
            print("Something is selected. {0}".format(rv.data[index]))
#            App.get_running_app().testLinkages()
            App.get_running_app().selected_title = rv.data[index]['text']
            App.get_running_app().selected_index = App.get_running_app(). \
                                        extracted_elements_dictionary['title'].\
                                        index(App.get_running_app().selected_title)
            # Update the details
            App.get_running_app()._updateDetails()
            
            print(index)
            print(rv.data)
        else:
            print("Selection removed from: {0}".format(rv.data[index]))
#            App.get_running_app().testLinkages()
            


class MainWindow(BoxLayout):
    '''
    This defines the main window. The main window should have three parts: 
        1. a left pane with buttons and text inputs;
        2. a central pane where certain text output should be displayed;
        3. a right pane with buttons for working with the central pane
        
    Uses horizontal box layout to contain the three panes
    '''
    leftpane = ObjectProperty(None)

class LeftPane(BoxLayout):
    '''
    The left pane contains a text input for the start date; a text input for the
    end date; and three buttons. Button 1 downloads the data for the given period
    from the internet; Button 2 gets the data from the database; Button 3 stores
    any unstored data into the database.
    '''
    from_dt = ObjectProperty(None)
    to_dt = ObjectProperty(None)

class CentralPane(RecycleView):
    '''
    The central pane has two text output sub-panes, both containing some text.
    The first is an article title; the second - its respective date.
    
    The central pane should be Recycle View.
    '''
#    rv_layout = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(CentralPane, self).__init__(**kwargs)
        self.data = kwargs.get('data', 'nothing found')

class RightPane(GridLayout):
    '''
    The right pane contains three buttons: 
        Button 1 assigns a "keep" flag to a selected item in the central pane;
        Button 2 assings a "discard" flag to a selected item in the central pane;
    '''
    pass
