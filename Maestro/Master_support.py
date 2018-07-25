# -*- coding: utf-8 -*-
import rospy
import time
import math as mt
import numpy as np
import os
from geometry_msgs.msg import Point
from Maquina_edos import TrafficSignal
from Maquina_edos2 import TrafficLight
TF=TrafficSignal()
TL=TrafficLight()
# Global constants
STOP_VELOCITY = 0
CURRENT = -1
NO_LINE = -1
REVERSE = -1
NO_OBSTACLE = -1.0

LANE_DRIVING = 0
INTER_APPROACHING = 1
WATTING = 2
#FOLLOWING = 3
MOVING_LEFT = 3
PASSING = 4
MOVING_RIGHT = 5

# Task names definition
task_names = {
        LANE_DRIVING:'Lane driving',
        INTER_APPROACHING: 'Intersection approaching',
        WATTING: 'Watting',
#        FOLLOWING: 'Following',
        MOVING_LEFT: 'Moving to left lane',
        PASSING: 'Passing obstacle',
        MOVING_RIGHT: 'Returning right lane'
        }

# Misc. functions definition

def AjusteVeinte(velocidadactual):
    if velocidadactual > -200:
	velocidadactual -= 5
    elif velocidadactual < -200:
	velocidadactual += 5

    return velocidadactual


def AjusteCincuenta(velocidadactual):
    if velocidadactual > -400:
	velocidadactual -= 5
    elif velocidadactual < -400:
	velocidadactual += 5

    return velocidadactual

def AjusteSetenta(velocidadactual):

    if velocidadactual > -600:
	velocidadactual -= 5

    elif velocidadactual < -600:
	velocidadactual += 5

    return velocidadactual

def AjusteCien(velocidadactual):
    if velocidadactual > -800:
	velocidadactual -= 5
    return velocidadactual


def calculate_speed(vel_decreasing_factor,
                    dist_to_line):
    """
    Calculates the PWM speed value accordingly
    to the distace to the intersection line.
    """

    tmp = \
        vel_decreasing_factor * dist_to_line

    if tmp > -200:
        return -200
    else:
        return tmp

def calculate_steering(pwm_steering_center,
                       steering_change_factor,
                       line_angle):
    """
    Calculates the PWM steering accordingly
    to the line angle in the intersections.
    """

    calculated_steering = \
       int(steering_change_factor * line_angle)


    return pwm_steering_center + calculated_steering

def speed_saturation(current_speed, calculated_speed):
    """
    Saturates the PWM speed smoothly.
    """
    if calculated_speed > current_speed:
        current_speed += 10
    elif calculated_speed < current_speed:
        current_speed -= 5

    return current_speed

def steering_saturation(current_steering, calculated_steering):
    """
    Saturates the PWM speed smoothly.
    """
    if calculated_steering > current_steering:
        current_steering += 1
    elif calculated_steering < current_steering:
        current_steering -= 1

    return current_steering


# Classes definition
class Task:
    """
    This class contains the task descriptors.
    """

    name = None
    ID = None

    def __init__(self, task_identifier):
        self.name = task_names[task_identifier]
        self.ID = task_identifier


class Master:
    """
    This class contains the master state variables
    that are taken into consideration for the state
    assigning and solving steps.
    """

    pwm_steering_center = None
    crossing_speed = None
    vel_decreasing_factor = None
    steering_change_factor = None
    max_dist_to_line = None
    min_dist_to_line = None
    dist_to_keep = None

    lane_speed = None
    lane_steering = None

    dist_to_line = None
    line_angle = None

    number_obstacles = None
    obstacles = None
    max_waiting_time = None

    tl_status= None
    ts_status= None

    task_pile = None
    count = None
    current_speed = None
    current_steering = None

    passing_enabled = None
    lights = None


    def __init__(self,
                 PWM_STEERING_CENTER,
                 CROSSING_SPEED,
                 VEL_DECREASING_FACTOR,
                 STEERING_CHANGE_FACTOR,
                 MAX_DIST_TO_LINE,
                 MIN_DIST_TO_LINE,
                 DIST_TO_KEEP,
                 MAX_WAIT_TIME,
                 PASSING_ENABLED,
                 task):

        self.pwm_steering_center = \
            PWM_STEERING_CENTER
        self.crossing_speed = CROSSING_SPEED
        self.vel_decreasing_factor = \
            VEL_DECREASING_FACTOR
        self.steering_change_factor = \
            STEERING_CHANGE_FACTOR
        self.max_dist_to_line = \
            MAX_DIST_TO_LINE
        self.min_dist_to_line = \
            MIN_DIST_TO_LINE
        self.lane_speed = -200
        self.lane_steering = \
            PWM_STEERING_CENTER
        self.dist_to_line = NO_LINE
        self.line_angle = 0
        self.number_obstacles = 0
        self.dist_to_keep = DIST_TO_KEEP
        self.obstacles = []
        self.max_waiting_time = MAX_WAIT_TIME
        self.task_pile = []
	    self.tl_status = None
     	self.ts_status = None
        self.count = 0
        self.current_speed = 0
        self.current_steering = \
            PWM_STEERING_CENTER
        self.passing_enabled = PASSING_ENABLED
        self.add_task(task)



    def add_task(self, task):
        """
        Adds a new task to the pile.
        """
        self.task_pile.append(task)

    def remove_task(self, task_index):
        """
        Remove the last task from the pile.
        """
        if len(self.task_pile) > 1:
            self.task_pile.pop(task_index)

    def get_current_task(self):
        """
        Get the indicated task from the pile.
        """
        return self.task_pile[CURRENT]

    def task_assigner(self):
        """
        Verifies the current enviroment status and
        checks whether is necesary, or not, to add
        a new task to the pile.
        """

        # Get the current task
        current_task = self.get_current_task()
        # First, checks if a close intersection exists.
        if self.dist_to_line > 0:

            # Checks current task to avoid intersection
            # routine duplication
            if (current_task.ID != INTER_APPROACHING
                and current_task.ID != WATTING):

                # Adds intersection routine
                self.add_task(Task(WATTING))
                self.add_task(Task(INTER_APPROACHING))

        # If not, check for obstacles
        elif self.dist_to_line == NO_LINE:


            # Checks the number of detected obstacles.
            if self.number_obstacles > 0:


                # Checks each obstacle info
                for obstacle in self.obstacles:

                    # Obstacle in front detected while driving
                    if (obstacle.x > 330.0) or (obstacle.x < 30.0):

                        if current_task.ID == LANE_DRIVING:

                            # Adds following routine
#                            self.add_task(Task(FOLLOWING))
#                            break

#                        if ((current_task.ID == FOLLOWING)
#                            and (self.count > self.max_waiting_time)):
#
#                            rospy.loginfo(self.max_waiting_time)
                            # Resets count
#                            self.count = 0

                            #self.remove_task(CURRENT)


                            # Adds passing obstacle routine
                            self.add_task(Task(MOVING_RIGHT))
                            self.add_task(Task(PASSING))
                            self.add_task(Task(MOVING_LEFT))
#                            break



    def task_solver(self):
        """
        Evaluates the current task to set the speed
        and steering policies accordingly.
        """
        # Get the current task
        current_task = self.get_current_task()
       # print('TS: Current task: ' + str(current_task.name))

        # Evaluates and tires to sove the current
        # task, setting the speed and steering
        # policies accordingly.

        # Lane driving case
        if current_task.ID == LANE_DRIVING: #Ajustar velocidades segun la señal de transito
	    TF.on_event(self.ts_status)

	    #print(self.current_speed)
	    if str(TF.state) == 'Cincuenta':
		print('Se�al de L�mite de velocidad 50...')
		print('Ajustando velocidad...')
		nuevavelocidad = AjusteCincuenta(self.current_speed)
		self.current_speed = nuevavelocidad
		self.current_steering = self.lane_steering
		self.lights = 'diL'
	    elif str(TF.state) == 'Setenta':
		print('Se�al de L�mite de velocidad 70...')
		print('Ajustando velocidad...')
		nuevavelocidad = AjusteSetenta(self.current_speed)
		self.current_speed = nuevavelocidad
		self.current_steering = self.lane_steering
		self.lights = 'diL'
	    elif str(TF.state) == 'Cien':
		print('Señal de Límite de velocidad 100...')
		print('Ajustando velocidad...')
		nuevavelocidad = AjusteCien(self.current_speed)
		self.current_speed = nuevavelocidad
		self.current_steering = self.lane_steering
		self.lights = 'diL'
	    elif str(TF.state) == 'Veinte':
	    	print('Señal de Límite de Velocidad 20...')
	    	print('Ajustando velocidad...')
	    	nuevavelocidad = AjusteVeinte(self.current_speed)
	    	self.current_speed = nuevavelocidad
	    	self.current_steering = self.lane_steering
	    	self.lights = 'diL'
	    else:
		self.current_speed= -200
                self.current_steering = self.lane_steering

        # Intersection approaching case
        elif current_task.ID == INTER_APPROACHING:

            # Checks if terminal conditions are met
            if (self.dist_to_line > 0  and #Subir distancia minima
                self.dist_to_line <= self.min_dist_to_line):
                self.current_speed = -400
                #self.current_steering = \
                #    calculate_steering(self.pwm_steering_center,
                #                       self.steering_change_factor,
                #                       self.line_angle)
		print(self.line_angle)
		self.current_steering = self.lane_steering
                self.lights = 'fr'

                    # Removes current task from pile
                self.remove_task(CURRENT)

            # if not, continue with the task policies
            else:

                # Sets speed and steeering policies
                self.current_speed = -300
                self.current_steering = \
                    calculate_steering(self.pwm_steering_center,
                                       self.steering_change_factor,
                                       self.line_angle)
		#self.current_steering = self.lane_steering
                self.lights = 'stop'

        # Watting case
        elif current_task.ID == WATTING:

            # Intersection line approaching case
            if (self.dist_to_line > 0):

                if (self.dist_to_line < self.min_dist_to_line):
		    TL.on_event(self.tl_status)
		    #print(self.current_speed)
		    if self.tl_status == 'verde':
		  #  if str(TL.state) == 'Verde':
			print('Semaforo verde')
			self.current_speed = -300
			#self.current_steering = \
			#	calculate_steering(self.pwm_steering_center,
			#			   self.steering_change_factor,
			#		 	   self.line_angle)
			#self.current_steering = self.lane_steering
                 #       self.lights = 'fr'
		    elif self.tl_status == 'rojo':
		  #  elif str(TL.state) == 'Rojo':
			print('Semaforo rojo')
			self.current_speed = 0
		#	self.lights = 'fr'
		  #  elif str(TL.state) == 'Amarillo':
		    elif self.tl_status == 'amarillo':
			print('Semaforo amarillo')
			self.current_speed = -200
			#self.current_steering = \
			#	calculate_steering(self.pwm_steering_center,
			#			   self.steering_change_factor,
			#			   self.line_angle)
			#self.current_steering = self.lane_steering


		#	self.lights = 'fr'

		    elif self.ts_status == 'Alto':
			print("Señal de alto, deteniendose...")
			self.current_speed = 0
			time.sleep(5)
			self.current_speed = -300
			#self.current_steering = \
			#	calculate_steering(self.pwm_steering_center,
			#			   self.steering_change_factor,
			#			   self.line_angle
			self.current_steering = self.lane_steering
                     #   self.lights = 'fr'


                    # Set policies
                    #self.current_speed = -200
                    #self.current_steering = \
                     #   calculate_steering(self.pwm_steering_center,
#                                           self.steering_change_factor,
 #                                          self.line_angle)
                   # self.lights = 'diL'

                else:
                    # Set policies
		    self.current_speed = self.crossing_speed
		    self.current_steering = self.lane_steering

                    # Removes current task from pile
                    self.remove_task(CURRENT)

            # Intersection end case
            else:

                # Kill LaneDetection node to restart it
                os.system('rosnode kill LaneDetection')

                # Set policies
                self.current_speed = -200
                self.current_steering = self.lane_steering
                self.lights = 'diL'

                # Removes current task from pile
                self.remove_task(CURRENT)

        #elif current_task.ID == FOLLOWING:

            # If no obstacles detected
        #    if self.number_obstacles == 0:

                # Set policies
        #        self.current_speed = -150
        #        self.current_steering = self.lane_steering
        #        self.lights = 'diL'

                # Reset count

                # Removes current task from pile
        #        self.remove_task(CURRENT)


            # Checks distance to each obstacle
        #    else:

        #        for obstacle in self.obstacles:

                    # No obstacle in front, finish task
        #            if (obstacle.x > 30.0 and obstacle.x < 330.0):

                        # Set policies
        #                self.current_speed = -150
        #                self.current_steering = self.lane_steering
        #                self.lights = 'diL'

                        # Reset count
        #                self.count = 0

                        # Removes current task from pile
        #                self.remove_task(CURRENT)
        #                break

                    # Following case
        #            elif (obstacle.y > (self.dist_to_keep + 10)):
        #                self.count = 0
                        # Set policies
        #                self.current_speed = \
        #                    following_speed(self.vel_decreasing_factor,
        #                                    self.dist_to_keep,
        #                                    obstacle.y)
        #                self.current_steering = self.lane_steering
        #                self.lights = 'stop'
        #                break

                    # Reverse case
        #            elif (obstacle.y < (self.dist_to_keep - 10)):
#
#                        self.count = 0
#                        # Set policies
#                        self.current_speed = \
#                            following_speed(self.vel_decreasing_factor,
#                                            self.dist_to_keep,
#                                            obstacle.y)
#                        self.current_steering = self.lane_steering
#                        self.lights = 're'
#                        break

                    # Waitting case
#                    else:

                        # Counting
#                        if self.passing_enabled == True:
#                            self.count += 1
#                        time.sleep(0.5)
#                        rospy.loginfo("Waiting, counting... %i" % self.count)

                        # Set policies
#                        self.current_speed = 0
#                        self.current_steering = self.lane_steering
#                        self.lights = 'diL'
#                        break

        elif current_task.ID == MOVING_LEFT:

            for obstacle in self.obstacles:

                # Ostacle in front, move to left lane
                if (obstacle.x < 27.0) or (obstacle.x > 330.0):

                    # Set policies
                    self.current_speed = -250
                    #self.current_steering = 150
                    self.current_steering = 140
                    self.lights = 'le'
                    break

                # On left lane
                elif (obstacle.x > 27.0) and (obstacle.x < 120.0):

                    # Kill LaneDetection node to restart it
                    os.system('rosnode kill LaneDetection')

                    # Set policies
                    self.current_speed = self.lane_speed
                    self.current_steering = self.lane_steering
                    self.lights = 'diL'

                    # Removes current task from pile
                    self.remove_task(CURRENT)
                    break

        elif current_task.ID == PASSING:

            for obstacle in self.obstacles:

                # Ostacle passed, finish task
                if (obstacle.x > 70.0) and (obstacle.x < 130.0):

                    # Set policies
                    self.current_speed = -250
                    self.current_steering = 50
                    self.lights = 'ri'

                    # Removes current task from pile
                    self.remove_task(CURRENT)
                    break

                # Drive left lane
                else:

                    # Set policies
                    self.current_speed = self.lane_speed
                    self.current_steering = self.lane_steering
                    self.lights = 'diL'



        elif current_task.ID == MOVING_RIGHT:

            for obstacle in self.obstacles:
#
                # Ostacle passed, finish task
                if (((obstacle.x > 110.0) and (obstacle.x < 180.0))
                    and obstacle.y > 53.0):

                    # Kill LaneDetection node to restart it
                    os.system('rosnode kill LaneDetection')

                    # Set policies
                    self.current_speed = self.lane_speed
                    self.current_steering = self.lane_steering
                    self.lights = 'diL'

                    # Removes current task from pile
                    self.remove_task(CURRENT)
                    break

                # Return right lane
                elif (obstacle.x > 80) and (obstacle.x < 110):

                    # Set policies
                    self.current_speed = -300
                    self.current_steering = 40
                    self.lights = 'ri'




    def run(self):
        """
        Executes the task assigner and solver
        to process the received data.
        """

        # Task assigner
        self.task_assigner()
        # Task solver
        self.task_solver()
