# -*- coding: utf-8 -*-

class State(object):

    def __init__(self):

        print 'Processing current state:', str(self)
        print 'Executing routine...'

    def on_event(self, event):

        pass

    def __repr__(self):

        return self.__str__()

    def __str__(self):

        return self.__class__.__name__

class NoLight(State): #Estos son los estados

    def on_event(self, event):
        if event == 'rojo':
            return Rojo()

        elif event == 'amarillo':
            return Amarillo()

        elif event == 'verde':
            return Verde()

        return self

class Rojo(State):

    def on_event(self,event):

        if event == 'verde':
            return Verde()
	elif event == 'amarillo':
	    return Amarillo()	
        return self

class Amarillo(State):
    def on_event(self,event):

        if event == 'rojo':
            return Rojo()
	elif event == 'verde':
	    return Verde()
	elif event == 'None':
	    return NoLight()
        return self

class Verde(State):
    def on_event(self,event):
        if event == 'amarillo':
            return Amarillo()
	elif event == 'rojo':
	    return Rojo()
	elif event == 'None':
	    return NoLight()
        return self

class TrafficLight(object):
    def __init__(self):
        self.state= NoLight() #Este es el estado con el que comienza

    def on_event(self, event):
        self.state = self.state.on_event(event) #Estas son las transiciones
