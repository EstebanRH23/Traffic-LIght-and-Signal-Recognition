# -*- coding: utf-8 -*-
import matplotlib.image as mpimg
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from Funcionesapoyo import *
import pickle
import time
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



from sklearn.model_selection import validation_curve
Stop=glob.glob('Datasets/48x90/Alto/*.png')
Setenta=glob.glob('Datasets/48x90/Setenta/*.png')
Cincuenta=glob.glob('Datasets/48x90/Cincuenta/*.png')
Cien=glob.glob('Datasets/48x90/Cien/*.png')
Veinte=glob.glob('Datasets/48x90/Veinte/*.png')
Background=glob.glob('Datasets/48x90/Background/*.png')

def extract_features(imgs, orient=9, pix_per_cell=8, cell_per_block=3, vis=False, feature_vector=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        feature_image = np.copy(gray)

        hog_features = get_hog_features(feature_image , orient,
                            pix_per_cell, cell_per_block, vis, feature_vector)

        hist_features = color_hist(hsv_image, 32)

        # Append the new feature vector to the features list
        file_features.append(hog_features)
        file_features.append(hist_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

Stop_features = extract_features (Stop, orient= 9, pix_per_cell=8, cell_per_block=3)
Setenta_features = extract_features (Setenta, orient=9, pix_per_cell= 8, cell_per_block=3)
Cincuenta_features = extract_features (Cincuenta, orient= 9, pix_per_cell=8, cell_per_block=3)
Cien_features = extract_features (Cien, orient= 9, pix_per_cell=8, cell_per_block=3)
Veinte_features = extract_features (Veinte, orient= 9, pix_per_cell=8, cell_per_block=3)
Background_features = extract_features (Background, orient= 9, pix_per_cell=8, cell_per_block=3)

X = np.vstack((Cien_features, Setenta_features, Cincuenta_features, Veinte_features, Stop_features, Background_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

a=np.empty(len(Cien_features))
a.fill(0)
b=np.empty(len(Setenta_features))
b.fill(1)
c=np.empty(len(Cincuenta_features))
c.fill(2)
d= np.empty(len(Veinte_features))
d.fill(3)
e= np.empty(len(Stop_features))
e.fill(4)
f= np.empty(len(Background_features))
f.fill(5)

# Define the labels vector
y = np.hstack((a,b,c,d,e,f))


X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.35, random_state=0)
print('Feature vector length:', len(X_train[0]))
svc= LinearSVC(C=0.0001,multi_class='crammer_singer') #Hay que probar el kernel para reportar en los resultados, y medir tiempos de entrenamiento.

svc.fit(X_train, y_train)

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
joblib.dump(svc, "Senales75x75.pkl", compress=3)
