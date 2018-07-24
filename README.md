# Traffic-LIght-and-Signal-Recognition
Repository with code, datasets and documentation of the terminal project named "Traffic Light and Traffic Signal Recognition applied to autonomous navigation of mobile robot"

# Summary
There are three main parts of the code that run in your computer with the help of a conda environment:

- SVM_final.py : Code with the training parameters and datasets of a Support Vector Machine with linear kernel.
- Funcionesapoyo.py : Miscellaneous functions used in the execution of the algorithm. (Sliding window algorithm, heat map, HOG parameters, etc.)
- TrafficSign_Detector: Code where the image publish by the camera in the vehicle is imported and where the color segmentation part is realized to finally publish the topic with the bounding boxes marking the traffic signals.
