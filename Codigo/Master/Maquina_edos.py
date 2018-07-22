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

class NoSignal(State): #Estos son los estados

    def on_event(self, event):

        if event == 'Cien':
            return Cien()
	   	
        elif event == 'Setenta':
	    return Setenta()

        elif event == 'Cincuenta':
            return Cincuenta()
	
	elif event == 'Veinte':
	    return Veinte()	

        return self

class Cien(State):

    def on_event(self,event):
        if event == 'Setenta':
            return Setenta()
        elif event == 'Cincuenta':
            return Cincuenta()
	elif event == 'Veinte':
	    return Veinte()
        return self

class Setenta(State):
    def on_event(self,event):
	
        if event == 'Cien':
            return Cien()
        elif event == 'Cincuenta':
            return Cincuenta()
	elif event == 'Veinte':
	    return Veinte()
        return self

class Cincuenta(State):
    def on_event(self,event):
        if event == 'Setenta':
            return Setenta()
        elif event == 'Cien':
            return Cien()
	elif event == 'Veinte':
	    return Veinte()
        return self

class Veinte(State):
    def on_event(self,event):
	if event == 'Setenta':
	    return Setenta()
	elif event == 'Cien':
	    return Cien()
	elif event == 'Cincuenta':
	    return Cincuenta() 
	return self

class TrafficSignal(object):
    def __init__(self):
        self.state= NoSignal() #Este es el estado con el que comienza

    def on_event(self, event):
        self.state = self.state.on_event(event) #Estas son las transiciones
