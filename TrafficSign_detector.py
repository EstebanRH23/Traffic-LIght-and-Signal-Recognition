
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
from std_msgs.msg import Int16, String
import time
from sklearn.externals import joblib
from Funcionesapoyo import *
from SVM_final import X_scaler

def image_callback(ros_data):

    e1 = cv2.getTickCount()
    np_arr= np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    crop = image_np[0:200,200:640]
    color = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0,100,150])
    upper_red = np.array([15,200,200]) #Señales de transito

    lower_stop = np.array([0,0,110])
    upper_stop = np.array([180,40,130]) ##Stop

    msk_r = cv2.inRange(hsv, lower_red, upper_red)
    msk_stop = cv2.inRange(hsv, lower_stop, upper_stop)
    stop=cv2.countNonZero(msk_stop)

    blur_r = cv2.GaussianBlur(msk_r,(5,5),5)
    circles_r = cv2.HoughCircles( blur_r , cv2.HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles_r is not None:
        circles_r = np.round(circles_r[0, :]).astype("int")
        radio = circles_r[0][2]
        if radio >= 15:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

            svc = joblib.load("Senales75x75.pkl")
            heat1= np.zeros_like(color[:,:,0]).astype(np.float)
            heat2= np.zeros_like(color[:,:,0]).astype(np.float)
            heat3= np.zeros_like(color[:,:,0]).astype(np.float)
            heat4= np.zeros_like(color[:,:,0]).astype(np.float)
            heat5= np.zeros_like(color[:,:,0]).astype(np.float)

            windows= slide_window(color, x_start_stop=[None, None], y_start_stop=[None,None],
                                xy_window=(64,120), xy_overlap=(0.75, 0.75))

            hot_windows,hot_windows2, hot_windows3, hot_windows4, hot_windows5= search_windows(color, windows, svc, X_scaler,
                                         orient=9, pix_per_cell=8, cell_per_block=3)

            # Image without heatmaṕ
            window_img = draw_boxes(color, hot_windows, hot_windows2, hot_windows3, hot_windows4, color=(255,0,0), color2=(0,255,0), color3=(0,0,255), color4=(255,255,0), thick=3)
            # Heatmap
            heat1= add_heat(heat1,hot_windows)
            heat2= add_heat(heat2,hot_windows2)
            heat3= add_heat(heat3,hot_windows3)
            heat4= add_heat(heat4,hot_windows4)
            heat5= add_heat(heat5,hot_windows5)

            heat1= apply_threshold(heat1, 0.99)
            heat2= apply_threshold(heat2, 0.99)
            heat3= apply_threshold(heat3, 0.99)
            heat4= apply_threshold(heat4, 0.99)
            heat5= apply_threshold(heat5, 0.99)

            heatmap = np.clip(heat1, 0, 255)
            heatmap2 = np.clip(heat2,0, 255)
            heatmap3 = np.clip(heat3,0, 255)
            heatmap4= np.clip(heat4, 0, 255)
            heatmap5= np.clip(heat5,0,255)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            labels2 = label(heatmap2)
            labels3= label(heatmap3)
            labels4= label(heatmap4)
            labels5= label(heatmap5)

            draw_img, s = draw_labeled_bboxes(color, labels, labels2, labels3, labels4, labels5)

        else:
            draw_img = color
            window_img = color
            s="None"
            print("Maybe a false possitive")



    elif stop>=1000:
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        svc = joblib.load("Senales75x75.pkl")
        heat1= np.zeros_like(color[:,:,0]).astype(np.float)
        heat2= np.zeros_like(color[:,:,0]).astype(np.float)
        heat3= np.zeros_like(color[:,:,0]).astype(np.float)
        heat4= np.zeros_like(color[:,:,0]).astype(np.float)
        heat5= np.zeros_like(color[:,:,0]).astype(np.float)

        windows= slide_window(color, x_start_stop=[None, None], y_start_stop=[None,None],
                            xy_window=(64,120), xy_overlap=(0.75, 0.75))

        hot_windows,hot_windows2, hot_windows3, hot_windows4, hot_windows5= search_windows(color, windows, svc, X_scaler,
                                     orient=9 , pix_per_cell=8, cell_per_block=3)
        # Image without heatmaṕ
        window_img = draw_boxes(color, hot_windows, hot_windows2, hot_windows3, hot_windows4, color=(255,0,0), color2=(0,255,0), color3=(0,0,255), color4=(255,255,0), thick=3)        #lower_red = np.array([0,150,150])
     

        # Heatmap

        heat1= add_heat(heat1,hot_windows)
        heat2= add_heat(heat2,hot_windows2)
        heat3= add_heat(heat3,hot_windows3)
        heat4= add_heat(heat4,hot_windows4)
        heat5= add_heat(heat5,hot_windows5)

        heat1= apply_threshold(heat1, 0.99)
        heat2= apply_threshold(heat2, 0.99)
        heat3= apply_threshold(heat3, 0.99)
        heat4= apply_threshold(heat4, 0.99)
        heat5= apply_threshold(heat5, 0.99)

        heatmap = np.clip(heat1, 0, 255)
        heatmap2 = np.clip(heat2,0, 255)
        heatmap3 = np.clip(heat3,0, 255)
        heatmap4= np.clip(heat4, 0, 255)
        heatmap5= np.clip(heat5,0,255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        labels2 = label(heatmap2)
        labels3= label(heatmap3)
        labels4= label(heatmap4)
        labels5= label(heatmap5)

        draw_img, s= draw_labeled_bboxes(color, labels, labels2, labels3, labels4, labels5)

    else:
        draw_img = color
        window_img = color
        s="None"

    e2 = cv2.getTickCount()
    final= CompressedImage()
    final.header.stamp = rospy.Time.now()
    final.format = 'jpeg'
    final.data = np.array(cv2.imencode('.jpg', draw_img)[1]).tostring()
    pub.publish(final)
    pubTS.publish(s)


def main():
    global pub, pubTS
    rospy.init_node('TS_detector', anonymous=True)
    rospy.Subscriber("/app/camera/rgb/image_raw/compressed", CompressedImage, image_callback, queue_size=1, buff_size=2**24)
    pub=rospy.Publisher('/TS_detector/compressed',CompressedImage, queue_size=1)
    pubTS=rospy.Publisher('/ROS_CONDA/TStatus', String , queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
