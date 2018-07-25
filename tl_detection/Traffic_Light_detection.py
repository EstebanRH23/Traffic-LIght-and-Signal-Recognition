#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2
import cv2.cv
from std_msgs.msg import Int16, String
import time
#import matplotlib.image as mpimg


def image_callback(ros_data):

    np_arr= np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    crop = image_np[0:200,200:640]
    color = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    #hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
    #hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    #nueva= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


              #mask= cv2.GaussianBlur(hsv,(5,5),0)
              #rojos #0-20 0 255 120 255

    lower_red = np.array([0,180,170])
    upper_red = np.array([15,250,250]) ##Los buenos
    #lower_red = np.array([0,100,150])
    #upper_red = np.array([15,200,200])
              #amarillo
    lower_yellow = np.array([11,200,150])
    upper_yellow = np.array([40,280,200])
              #verdes15250
    lower_green = np.array([30,80,100])
    upper_green = np.array([90,300,300])

    msk_r = cv2.inRange(hsv, lower_red, upper_red)
    msk_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    msk_g = cv2.inRange(hsv, lower_green, upper_green)

    r=cv2.countNonZero(msk_r)
    y=cv2.countNonZero(msk_y)
    g=cv2.countNonZero(msk_g)

    #print(r)
    #print(y)
    #print(g)

    blur_r = cv2.GaussianBlur(msk_r,(5,5),5)
    blur_y = cv2.GaussianBlur(msk_y,(5,5),5)
    blur_g = cv2.GaussianBlur(msk_g,(5,5),5)

    circles_r = cv2.HoughCircles( blur_r , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles_y = cv2.HoughCircles( blur_y , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles_g = cv2.HoughCircles( blur_g , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)

    #circles_r = cv2.HoughCircles( blur_r , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)
    #circles_y = cv2.HoughCircles( blur_y , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)
    #circles_g = cv2.HoughCircles( blur_g , cv2.cv.CV_HOUGH_GRADIENT, 1,20, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles_r is not None:
        circles_r = np.round(circles_r[0, :]).astype("int")
        if circles_r[0][2] <= 15 or circles_r[0][2] >= 10:
            print ("Semaforo rojo")
            cv2.circle(color,(circles_r[0][0],circles_r[0][1]),circles_r[0][2],(0,0,255),4)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(color, 'Rojo', (circles_r[0][0],50), font, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
            pubTL.publish("rojo")
    elif circles_y is not None:
        circles_y = np.round(circles_y[0, :]).astype("int")
        if circles_y[0][2] <= 20  or circles_y[0][2] >= 10:
            print ("Semaforo amarillo")
            cv2.circle(color,(circles_y[0][0],circles_y[0][1]),circles_y[0][2],(0,255,255),4)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(color, 'Amarillo', (circles_y[0][0],50), font, 0.8, (0, 255, 255), 3, cv2.LINE_AA)
            pubTL.publish("amarillo")
    elif circles_g is not None:
        circles_g = np.round(circles_g[0, :]).astype("int")
        if circles_g[0][2] <= 20 or circles_g[0][2] >= 10:
            print ("Semaforo verde")
            cv2.circle(color,(circles_g[0][0],circles_g[0][1]),circles_g[0][2],(0,255,0),4)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(color, 'Verde', (circles_g[0][0],50), font, 0.8, (0, 255, 0), 3, cv2.LINE_AA)
            pubTL.publish("verde")
    else:
	pubTL.publish('None')	

    #umbral=50
    #colorcito = (0,0,0)
    #if r>umbral:
    #    colorcito= (255,100,250)
    #    print("El semaforo es rojo")
    #elif y>umbral:
    #    colorcito= (100,50,250)
    #    print("El semaforo es amarillo")
    #elif g>umbral:
    #    colorcito= (200,240,255)
    #    print("El semaforo es verde")
    #else:
    #    color= (255,255,255)
    #    print("El semaforo esta apagado")

    final= CompressedImage()
    final.header.stamp = rospy.Time.now()
    final.format = 'jpeg'
    final.data = np.array(cv2.imencode('.jpg', color)[1]).tostring()
    pub.publish(final)



def main():
    global pub, pubTL
    rospy.init_node('TL_detector', anonymous=True)
    rospy.Subscriber("/app/camera/rgb/image_raw/compressed", CompressedImage, image_callback, queue_size=1, buff_size=2**24)
    pub=rospy.Publisher('/TL_detector/compressed',CompressedImage, queue_size=1)
    pubTL=rospy.Publisher('/TL_detector/Estado',String,queue_size=1)
    rospy.spin()


if __name__ == '__main__':
    main()
