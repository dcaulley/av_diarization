#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:21:40 2019

@author: dcaulley: Written by Desmond Caulley
"""

import os,sys
import pickle
import argparse

import numpy as np

parser = argparse.ArgumentParser(description = "creates text times when faces are IDed")

parser.add_argument('--faces_file',  type=str,   default='',    help='eg. file with IDed faces stored as rectanglular boxes')
parser.add_argument('--output_file', type=str,   default='',    help='file to store start and end times of faces')
parser.add_argument('--fps',         type=float, default=25,    help='fps of video used to create face_file')


opt = parser.parse_args();


output_file = sys.argv[2]


with open(opt.faces_file, 'rb') as fil:
    faces = pickle.load(fil)  
    
frames_IDed = []
for f, face in enumerate(faces):
    if face != []:
        frames_IDed.append(f)
    
    
frame_dur=1/opt.fps

start_end_table =[];
start_time = round(frames_IDed[0]*frame_dur, 2)
      
for f in range(len(frames_IDed)):
    try:
        if frames_IDed[f+1] > frames_IDed[f] + 10:
            end_time = round(frames_IDed[f]*frame_dur, 2)  
            
            if end_time - start_time >= 1:
                start_end_table.append([start_time, end_time])
            start_time = round(frames_IDed[f+1]*frame_dur, 2)
    except:
        continue

if (frames_IDed[f]*frame_dur - start_time) > 1:
    start_end_table.append([start_time, frames_IDed[f]*frame_dur])

if (len(start_end_table) > 0):
    start_end_table = np.array(start_end_table).astype(str)
    
    start_end_table = np.insert(start_end_table, 0, ['start', 'end'], axis=0)
    np.savetxt(opt.output_file, start_end_table, fmt='%s')
