# Traffic-LIght-and-Signal-Recognition
Repository with code, datasets and documentation of the terminal project named "Traffic Light and Traffic Signal Recognition applied to autonomous navigation of mobile robot"

This repository was built with the help of [AutoModelCar CIC IPN](https://github.com/Conilo/automodelcar-cic-ipn)

**IMPORTANT: Before starting, make sure you have ROS, CONDA and all it's dependencies properly installed on your PC.
Otherwise, visit the [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) and [CONDA Tutorials](https://conda.io/docs/user-guide/tutorials/index.html)**

# Summary
There are three main parts of the code that run in your computer with the help of a conda environment:

- SVM_final.py : Code with the training parameters and datasets of a Support Vector Machine with linear kernel.
- Funcionesapoyo.py : Miscellaneous functions used in the execution of the algorithm. (Sliding window algorithm, heat map, HOG parameters, etc.)
- TrafficSign_Detector: Code where the image publish by the camera in the vehicle is imported and where the color segmentation part is realized to finally publish the topic with the bounding boxes marking the traffic signals.

## Cloning the repository
Create a new folder named "TS_detection" or whatever you want
> mkdir TS_detection

Access the folder 
> cd TS_detection

In order to start working with this code, please first clone the repository on the folder you previously created by typing:
> git clone https://github.com/EstebanRH23/Traffic-LIght-and-Signal-Recognition.git

## Running the code

First of all, make sure you have established communication with the car, with the command:

> ssh root@192.168.43.102

If you have any doubts about the communication with the car, you can check the documentation in [AutoModelCar CIC IPN](https://github.com/Conilo/automodelcar-cic-ipn) or [Automodelcar](https://github.com/AutoModelCar).

Parallel, you need to have Conda running in your machine with the purpose of source the environment called "", this environment have all the necessary to execute the code withouth problems.

You can enable the environment in your machine with:
> Source activate "Name of the environment"

Enter to the folder where you downloaded the repository:
> python TrafficSign_detector.py

Finally you can disable the environment with the command:
> Source deactivate "Name of the environment"

## Contact 

If you need more information about this code, please contact:

- Project Supervisors: 
  - Erik Zamora Gómez (E-mail: ezamora1981@gmail.com)
  - Cesar Gerardo Bravo Conejo (E-mail: conilo@gmail.com)
  
- Project Author:
  - Esteban Iván Rojas Hernández (E-mail: rojasesteban23@gmail.com)
   
