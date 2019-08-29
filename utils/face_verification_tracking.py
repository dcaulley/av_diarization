#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:03:16 2019

@author: dcaulley: Desmond Caulley

This script compares faces in video to template_file face. Once face is identified
and verified, we track it using CSRT tracking. To make sure we are properly tracking
face at time step, we keep a tab on the histogram of the face and check histogram at each
iteration. If histogram doesn't match, we go back to searching and verifying face. 
"""


import cv2
import dlib
import argparse

import pickle
import numpy as np


import imutils
from imutils import face_utils
import scipy.spatial.distance as dist

import tensorflow as tf
from keras import backend as K
from scene_detector import find_scenes
from keras_vggface.vggface import VGGFace

parser = argparse.ArgumentParser(description = "face detection, verification, and tracking")

parser.add_argument('--video_file',     type=str,   default='',            help='input video file'   )
parser.add_argument('--template_file',  type=str,   default='',            help='input template face')
parser.add_argument('--output_scenes',  type=str,   default='scenes.pckl', help='pckl file to store output scenes')
parser.add_argument('--output_faces',   type=str,   default='faces.pckl',  help='pckl file to store output faces')
parser.add_argument('--scoring_type',   type=str,   default='cosine',      help='option: cosine vs. euclidean'      )
parser.add_argument('--threshold',      type=float, default=0.35,          help='threshold to compare vector distance: Euc=140')
parser.add_argument('--frame-width',    type=int,   default=500,           help='frame size shrinking to speed processing')


opt          = parser.parse_args();
frame_width  = opt.frame_width

template_faces = np.load(opt.template_file)

def face_verif(features):    
    scores = []
    u = features
    for v in template_faces.T:
        if opt.scoring_type == 'cosine':
            scores.append(dist.cosine(u, v))
        else:
            scores.append(dist.euclidean(u, v))
    
    scores.sort()
    return np.mean(scores)

def img_reformat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return img


# Configuring to use only 70% of GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.90
session = tf.Session(config=config)
K.set_session(session)


# Import VGGFace feature extractor
detector = dlib.get_frontal_face_detector()
resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')



# Scene Detector
frame_rate, frames_read, scene_list = find_scenes(opt.video_file)


# Scene_list_mod. Suppose video is one long frame.
# Break into 5 subframes. Since we skip to next frame
# after some number of misses
scene_list_mod = np.concatenate((scene_list, np.linspace(0, frames_read, 5)))
scene_list_mod = np.unique(scene_list_mod).astype(int)


# Video Proocessing Setup
cap = cv2.VideoCapture(opt.video_file)


# Adjusting frame size for faster processing
fw,  fh     = float(cap.get(3)), float(cap.get(4))
frame_width = int(fw) if fw < frame_width else frame_width


ratio  = frame_width/fw
fw, fh = frame_width, ratio*fh


# Initializing while loop
rect_color = (66,245,108)
verified_face = [ [] for i in range(1000000) ]
miss_times, tracking_face, check_found = 0, 0, 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = imutils.resize(frame, width=frame_width)
        
    #If we are not tracking a face, then try to detect one
    if not tracking_face or (int(cap.get(1)) in scene_list_mod):
        try:
            tracker, tracking_face = cv2.TrackerCSRT_create(), 0
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
                        
            for face in faces:
                (x, y, w, h) = face_utils.rect_to_bb(face)
                x = max(0, x); y = max(0, y)

                face = frame[y:y+h, x:x+w, :]
                face = img_reformat(face)
                
                features    = resnet50_features.predict(face)
                verif_score = face_verif(features)
    
                if verif_score < opt.threshold:
                    tracker.init(frame, (x, y, w, h))
                    
                    tracking_face, miss_times = 1, 0

                    # creating face pixel histogram. Helps with tracking.
                    hist_face = cv2.cvtColor(frame[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB)
                    hist_face = cv2.calcHist([hist_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    template_hist = cv2.normalize(hist_face, hist_face).flatten()
                    check_found   = 0
                    break
        except Exception as e:
            print(e)
            tracking_face = 0
            pass
                    
                
        if (miss_times != 0):
            for k in range(2):
                cap.read()
                
        miss_times += 1
        
        if miss_times >= 10:
            
            try:
                scene_list_mod = scene_list_mod[np.where(scene_list_mod > cap.get(1))[0]]
                end_point = min(int(scene_list_mod[0]), int(cap.get(1) + cap.get(5)*3))
                for k in range(int(end_point - cap.get(1))):
                    cap.read()
                miss_times = 0
                check_found += 1
            except Exception as e:
                print(e)
                miss_times = 0
                continue
            
    #Check if the tracker is actively tracking a region in the image
    if tracking_face:
        if not ret:
            break
        try:
            success, new_box = tracker.update(frame)
            
            # Check if histogram of face during tracking 
            # matches original histogram
            x, y, w, h = new_box
            x, y, w, h = int(max(0,x)), int(max(0,y)), int(w), int(h)


            if ((w < 2) and (h < 2)):
                tracking_face = 0
                continue

            hist_curr = cv2.cvtColor(frame[y:y+h, x:x+w, :], cv2.COLOR_BGR2RGB)
            hist_curr = cv2.calcHist([hist_curr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            curr_hist = cv2.normalize(hist_curr, hist_curr).flatten()
            
            hist_results = cv2.compareHist(curr_hist, template_hist, cv2.HISTCMP_CORREL)
            
            if hist_results > 0.7:
                p1 = (int(new_box[0]), int(new_box[1]))
                p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
                                
                verified_face[int(cap.get(1))].append([int(cap.get(1)), [y/fh, x/fw, (y+h)/fh, (x+w)/fw], verif_score])
    
            else:
                tracking_face = 0
        except Exception as e:
            print(e)
            tracking_face = 0
    
    if (not ret) or ((check_found >= 200) and (cap.get(1) > (cap.get(5)*60*5))):
        break
     
    
verified_face = verified_face[:int(cap.get(1))]

with open(opt.output_scenes, 'wb') as fil:
    pickle.dump([frame_rate, frames_read, scene_list], fil)
    
with open(opt.output_faces, 'wb') as fil:
    pickle.dump(verified_face, fil)
