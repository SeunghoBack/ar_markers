import cv2
import cv2.aruco as aruco
import numpy as np
import os

fPath = '/home/nearthlab/test_cpp/ComputerVision-Fun-Projects/Blur_detection/images/OksangPicture/'

def findArucoMarkers(img, markerSize = 6, totalMarkers=100, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    print(ids)

img = cv2.imread('/home/nearthlab/Pictures/Screenshot from 2021-11-25 15-35-34.png')
findArucoMarkers(img)
cv2.imshow('img',img)
cv2.waitKey(0)
