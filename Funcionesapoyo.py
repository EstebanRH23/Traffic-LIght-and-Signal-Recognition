import numpy as np
import cv2
from skimage.feature import hog
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis, feature_vector):

    features = hog(img,orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               visualise=vis, feature_vector=feature_vector)
    return features

def color_hist(img, nbins=32, bins_range=(0, 255)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def single_img_features(img, orient=9, pix_per_cell=8, cell_per_block=3, vis=False, feature_vector=True):

    #1) Define an empty list to receive features
    img_features = []
    feature_image = np.copy(img)
    feature_image_gray = cv2.cvtColor(feature_image, cv2.COLOR_RGB2GRAY)
    feature_image_hsv = cv2.cvtColor(feature_image, cv2.COLOR_RGB2HSV)

    hog_features = get_hog_features(feature_image_gray, orient,
                   pix_per_cell, cell_per_block, vis, feature_vector)

    hist_features = color_hist(feature_image_hsv, 32)

    #8) Append features to list
    img_features.append(hog_features)
    img_features.append(hist_features)
    #9) Return concatenated array of features
    return np.concatenate(img_features)

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 120), xy_overlap=(0.75, 0.75)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img,bboxes,bboxes2,bboxes3,bboxes4,color=(255,0,0),color2=(0,255,0),color3=(0,0,255),color4=(255,255,0), thick=3):
    # Make a copy of the imag
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    for bbox in bboxes2:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color2, thick)
    for bbox in bboxes3:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color3, thick)
    for bbox in bboxes4:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color4, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, orient=9,
                    pix_per_cell=8, cell_per_block=3):

    #1) Create an empty list to receive positive detection windows
    cnt0= 0
    cnt1= 0
    cnt2= 0
    cnt3= 0
    cnt4= 0
    cnt5= 0

    on_windows_alto = []
    on_windows_cien = []
    on_windows_cincuenta= []
    on_windows_veinte= []
    on_windows_setenta = []
    #2) Iterate over all windows in the list
    for window in windows:

        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (48, 90))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, orient=9, pix_per_cell=8,
                                       cell_per_block=3)

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        #print(prediction)
        #Inicializar variable que cuente la frecuencia con la que se repite cada uno de los elementos,
        #excepto el 4, si se repite 3 veces o mas, se hace el append de la ventana
        if prediction == 0:
            cnt0 = cnt0+1
            cnt0 == cnt0
            #print(cnt0)
            if cnt0 >= 1:
                print("Límite de velocidad 100")
                on_windows_cien.append(window)

        if prediction == 1:
            cnt1 = cnt1+1
            cnt1 == cnt1
            if cnt1 >= 2:
                print("Límite de velocidad 70")
                on_windows_setenta.append(window)

        if prediction == 2:
            cnt2 = cnt2+1
            cnt2 == cnt2
            if cnt2 >= 1:
                print("Límite de velocidad 50")
                on_windows_cincuenta.append(window)

        if prediction == 3:
            cnt3 = cnt3+1
            cnt3 == cnt3
            if cnt3 >= 2:
                print("Límite de velocidad 20")
                on_windows_veinte.append(window)

        if prediction == 4:
            cnt4 = cnt4+1
            cnt4 == cnt4
            if cnt4 >= 1:
                print("Señal de alto")
                on_windows_alto.append(window)

        if prediction == 6:
            cnt5 = cnt5+1
            cnt5 == cnt5
            if cnt5>=40:
                print("No action required")
        #8) Return windows for positive detections
    return on_windows_cien, on_windows_setenta, on_windows_cincuenta, on_windows_veinte, on_windows_alto

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for bbox in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, labels2, labels3, labels4, labels5):
    # Iterate through all detected cars
    s="None"

    for car_number in range(1, labels[1]+1):
#        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
         # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
#        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
        s= "Cien"

    #print(bbox)
    for car_number in range(1, labels2[1]+1):
        # Find pixels with each car_number label value
        nonzero2 = (labels2[0] == car_number).nonzero()
#    #    # Identify x and y values of those pixels
        nonzeroy2 = np.array(nonzero2[0])
        nonzerox2 = np.array(nonzero2[1])
    #    # Define a bounding box based on min/max x and y
        bbox2 = ((np.min(nonzerox2), np.min(nonzeroy2)), (np.max(nonzerox2), np.max(nonzeroy2)))
    #    # Draw the box on the image
        cv2.rectangle(img, bbox2[0] , bbox2[1], (0,255,0), 3)
        s= "Setenta"

        #print(bbox2)
    for car_number in range(1, labels3[1]+1):
        # Find pixels with each car_number label value
        nonzero3 = (labels3[0] == car_number).nonzero()
#    #    # Identify x and y values of those pixels
        nonzeroy3 = np.array(nonzero3[0])
        nonzerox3 = np.array(nonzero3[1])
    #    # Define a bounding box based on min/max x and y
        bbox3 = ((np.min(nonzerox3), np.min(nonzeroy3)), (np.max(nonzerox3), np.max(nonzeroy3)))
        # Draw the box on the image
        cv2.rectangle(img, bbox3[0] , bbox3[1], (255,0,255), 3)
        s= "Cincuenta"

    for car_number in range(1, labels4[1]+1):
        # Find pixels with each car_number label value
        nonzero4 = (labels4[0] == car_number).nonzero()
#    #    # Identify x and y values of those pixels
        nonzeroy4 = np.array(nonzero4[0])
        nonzerox4 = np.array(nonzero4[1])
    #    # Define a bounding box based on min/max x and y
        bbox4 = ((np.min(nonzerox4), np.min(nonzeroy4)), (np.max(nonzerox4), np.max(nonzeroy4)))
    #    # Draw the box on the image
        cv2.rectangle(img, bbox4[0] , bbox4[1], (150,100,255), 3)
        s= "Veinte"

    for car_number in range(1, labels5[1]+1):
        # Find pixels with each car_number label value
        nonzero5 = (labels5[0] == car_number).nonzero()
#    #    # Identify x and y values of those pixels
        nonzeroy5 = np.array(nonzero5[0])
        nonzerox5 = np.array(nonzero5[1])
    #    # Define a bounding box based on min/max x and y
        bbox5 = ((np.min(nonzerox5), np.min(nonzeroy5)), (np.max(nonzerox5), np.max(nonzeroy5)))
    #    # Draw the box on the image
        cv2.rectangle(img, bbox5[0] , bbox5[1], (0,255,255), 3)
        s= "Alto"

    return img, s
