#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:17:53 2019

@author: dcaulley: Desmond Caulley
"""

live_demo = 0

import sys
import numpy as np
import matplotlib.pyplot as plt


import cv2, imutils



from keras_vggface.vggface import VGGFace

face_file   = sys.argv[1]
input_vid   = sys.argv[2]
frame_width = int(sys.argv[3])
output_file = sys.argv[4]


faces_detected = np.load(face_file)
cap = cv2.VideoCapture(input_vid)
cap.set(cv2.CAP_PROP_FPS, 25)


resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')


def img_reformat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return img


frame_num = 0    
face_features = [] 
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        
        frame = imutils.resize(frame, width=frame_width)
        
        for f in np.where(faces_detected[0,:] == frame_num)[0]:
            
            x, y, w, h = faces_detected[1:,f]
            face = frame[y:y+h, x:x+w]
            face = img_reformat(face)
            
            features = resnet50_features.predict(face)

            face_features.append(features)
            
            if live_demo:
                plt.imshow(face)
                plt.show()
                
        frame_num +=1

    else:
        break
    
face_features = np.concatenate(face_features, axis = 0).T
np.save(output_file, face_features)

print('\nDone with Extracting Face Features Using VGGFace\n')