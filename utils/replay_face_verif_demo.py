#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:59:17 2019

@author: dcaulley: Desmond Caulley

This script is a live demo of the IDed face. It takes as input the fps-adjusted video and
faces_file generated after face search. The outcome is live video with box around the face
identified. You can choose to save the video with face ided by setting make_vid = 1
"""

import cv2
import pickle
import argparse

import imutils
import os, sys
import subprocess

import numpy as np
import scipy.spatial.distance as dist

parser = argparse.ArgumentParser(description = "replaying of tracked face adn making video of that")

parser.add_argument('--video_file',     type=str,   default='',            help='input video file'   )
parser.add_argument('--faces_file',     type=str,   default='',            help='input face file' )
parser.add_argument('--make_vid',       type=int,   default=1,             help='value=1 to save video file')

opt = parser.parse_args();

demo_vid_save = "eg_box_hist_mosalah.avi"


with open(opt.faces_file, 'rb') as fil:
    faces = pickle.load(fil)
    
frame_width = 500



# Video Proocessing Setup
cap         = cv2.VideoCapture(opt.video_file)
fw,  fh     = float(cap.get(3)), float(cap.get(4))
frame_width = int(fw) if fw < frame_width else frame_width

ratio  = frame_width/fw
fw, fh = frame_width, ratio*fh

rect_color = (66,245,108)

if opt.make_vid:
    fourcc     = cv2.VideoWriter_fourcc(*'XVID')
    vOut       = cv2.VideoWriter(demo_vid_save, fourcc, 25, (int(fw), int(fh)))


while True:
    ret, frame = cap.read()
    
    if not ret or (int(cap.get(1)) >= len(faces)):
        break
    
    frame = imutils.resize(frame, width=frame_width)
    
    if faces[int(cap.get(1))] != []:
        rect = faces[int(cap.get(1))][0][1]
        x, y = fw*rect[1], fh*rect[0]
        w, h = fw*(rect[3]  - rect[1]), fh*(rect[2] - rect[0])
        new_box = x, y, w, h
        p1 = (int(new_box[0]), int(new_box[1]))
        p2 = (int(new_box[0] + new_box[2]), int(new_box[1] + new_box[3]))
        cv2.rectangle(frame, p1, p2, rect_color, 2, 1)
    else:
        m =2
        pass
    if opt.make_vid:
        vOut.write(frame)
    else:
        cv2.imshow('TrackerA', frame)
        cv2.waitKey(1); 

if opt.make_vid:
    vOut.release()
