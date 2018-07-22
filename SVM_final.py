# -*- coding: utf-8 -*-
import matplotlib.image as mpimg
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from Funcionesapoyo import *
import pickle
import time
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc
#from sklearn import datasets
#from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



from sklearn.model_selection import validation_curve
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score

#from sklearn.model_selection import learning_curve
#from sklearn.model_selection import ShuffleSplit
#e1 = cv2.getTickCount()


#Checar que esta pasando con la clase Setenta
#Dedicarse a semáforos, cambiar de color a semaforos, entrenar, pintar, hacer uno nuevo, etc etc
#Segmentación de color
#Entrenar mas y mejores modelos
Stop=glob.glob('Datasets/48x90/Alto/*.png')
#Noventa=glob.glob('Datasets/75x75/Ochenta/*.png')
Setenta=glob.glob('Datasets/48x90/Setenta/*.png')
#Setenta=Setenta[51:150]
#Ochenta=glob.glob('Datasets/75x75/Ochenta2/*.png')
Cincuenta=glob.glob('Datasets/48x90/Cincuenta/*.png')
#Cincuenta.extend(glob.glob('Datasets/75x75/Cincuenta/*.png'))
Cien=glob.glob('Datasets/48x90/Cien/*.png')
#Cien.extend(glob.glob('Datasets/75x75/Cien/*.png'))
Veinte=glob.glob('Datasets/48x90/Veinte/*.png')
#Veinte.extend(glob.glob('Datasets/75x75/Veinte/*.png'))
Background=glob.glob('Datasets/48x90/Background/*.png')

#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                        n_jobs=1, train_sizes=np.linspace(0.1,1.0,5)):

#    plt.figure()
#    plt.title(title)
#    if ylim is not None:
#        plt.ylim(*ylim)
#    plt.xlabel("Training examples")
#    plt.ylabel("Score")
#    train_sizes, train_scores, test_scores = learning_curve(
#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#    train_scores_mean = np.mean(train_scores, axis=1)
#    train_scores_std = np.std(train_scores, axis=1)
#    test_scores_mean = np.mean(test_scores, axis=1)
#    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()
#
#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#             label="Training score")
#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation score")
#
#    plt.legend(loc="best")
#    return plt



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
#X = np.vstack((Stop_features,Setenta_features, Background_features)).astype(np.float64)
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
#y = label_binarize(y, classes=[0,1,2,3,4,5])
#n_classes = y.shape[1]
#n_classes = 6


X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.35, random_state=0)
print('Feature vector length:', len(X_train[0]))
svc= LinearSVC(C=0.0001,multi_class='crammer_singer') #Hay que probar el kernel para reportar en los resultados, y medir tiempos de entrenamiento.

#print(X_test.shape)
#print(X_train.shape)
#print(y_test.shape)
#print(y_train.shape)

#clf = LinearSVC()
#y_score = clf.fit(X_train, y_train).decision_function(X_test)

#print(y_score.shape)

#fpr = dict()
#tpr = dict()
##roc_auc = dict()
#for i in range(n_classes):print(f1_score(y_test, y_pred, average="macro"))
#print(precision_score(y_test, y_pred, average="macro"))
#print(recall_score(y_test, y_pred, average="macro"))

#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

#    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
#for i in range(n_classes):
#    plt.figure()
#    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()



#svc = SVC(kernel='poly')
#t=time.time()
svc.fit(X_train, y_train)
#y_pred = svc.predict(X_test)
#t2 = time.time()
#print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
#e2 = cv2.getTickCount()
#t = (e2 - e1)/cv2.getTickFrequency()
#print(t)
joblib.dump(svc, "Senales75x75.pkl", compress=3)
