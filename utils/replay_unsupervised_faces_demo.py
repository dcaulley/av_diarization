#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:59:17 2019

@author: dcaulley: Desmond Caulley
"""

import cv2
import pickle

import imutils
import os, sys
import subprocess

import numpy as np
import scipy.spatial.distance as dist


videoPath     =  '/Users/dcaulley/projects/audio_visual_diarization/exp_demo/video_processing/videoplayback/pyavi/video.avi'
faces_path    = '/Users/dcaulley/projects/audio_visual_diarization/exp_demo/video_processing/videoplayback/visual_diarization/videoplayback_person_faces.npy'
make_vid      = 1 #will save a video of the demonstration to view later if set to 1.
demo_vid_save = "demo_vid_unsup.avi"



track_faces = np.load(faces_path)    
frame_width = 500


# Video Proocessing Setup
cap = cv2.VideoCapture(videoPath)

fw,  fh     = float(cap.get(3)), float(cap.get(4))
frame_width = int(fw) if fw < frame_width else frame_width

ratio  = frame_width/fw
fw, fh = frame_width, ratio*fh

rect_color = []

for k in range(len(track_faces)):
    rect_color.append((np.random.randint(255), np.random.randint(255), np.random.randint(255)))
    
#cap.set(1, 100)
if make_vid:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vOut   = cv2.VideoWriter(demo_vid_save, fourcc, 25, (int(fw), int(fh)))

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame = imutils.resize(frame, width=frame_width)
    
    for k in range(len(track_faces)):
        if int(cap.get(1)) in track_faces[k][0]:
            loc = np.where(track_faces[k][0] == cap.get(1))[0][0]
            rect = track_faces[k][1][loc]
            x, y = fw*rect[1], fh*rect[0]
            w, h = fw*(rect[3]  - rect[1]), fh*(rect[2] - rect[0])
            new_box = x, y, w, h
            p1 = (int(new_box[0]), int(new_box[1]))
            p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
            cv2.rectangle(frame, p1, p2, rect_color[k], 2, 1)
        else:
            m = 2
            pass
    if make_vid:
        vOut.write(frame)
    else:
        cv2.imshow('TrackerA', frame)
        cv2.waitKey(1); 

if make_vid:
    vOut.release()