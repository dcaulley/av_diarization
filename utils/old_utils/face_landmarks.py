#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:12:13 2019

@author: dcaulley: Desmond Caulley
"""

"""
This script finds landmark features on faces and stores them as numpy file. 
The function is called as: face_landmarks.py input_file input_vid output_file

input_file is a numpy array with entries of frame and rectangle_box where face
is located
"""


import sys
import numpy as np

import cv2, dlib, imutils
from imutils import face_utils

live_demo = 0

input_file = sys.argv[1]
input_vid = sys.argv[2]
output_file = sys.argv[3]

frame_width = 500
lip_only = 0

faces_detected = np.load(input_file).T
landmark_pred = dlib.shape_predictor('conf/shape_predictor_194_face_landmarks.dat')


cap = cv2.VideoCapture(input_vid)
cap.set(cv2.CAP_PROP_FPS, 25)

frame_num = 0
landmarks_detected = []

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        for f in np.where(faces_detected[:,0] == frame_num)[0]:
            
            x, y, w, h = faces_detected[f,1:]
            rect = dlib.rectangle(x, y, x+w, y+h)
            
            shape = landmark_pred(gray, rect)
            
            landmarks_detected.append([frame, shape])
            
            shape = face_utils.shape_to_np(shape)
            
            if lip_only:
                shape = shape[152:,:]
            if live_demo:
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        if live_demo:
            cv2.waitKey(25)
            cv2.imshow("Output", frame)

        frame_num +=1



    else:
        break

landmarks_detected = np.array(landmarks_detected)
np.save(output_file, landmarks_detected)

print('\nDone with Landmark Detection\n')