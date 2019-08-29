#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:56:32 2019

@author: dcaulley: Desmond Caulley
"""


"""
Stitch video for given text file

This script goes through entire text files with start and end times in a 
directory and tries to stitched video files using those time stamps.
Each txt file with time-stamp comes from a unique video.

The script takes in directory with timestamp. And output file to store
concatenated video 
Eg ./video_stitching_ind time_stamp_dir/ output_filename
"""


import glob
import os, sys
import subprocess

import cv2
import numpy as np

input_vid      = sys.argv[1] #input video
timestamp_file = sys.argv[2] #.txt file with timestamps for input_vid

k = 0

tmp_dir = os.path.join(os.path.split(timestamp_file)[0], 'tmp_dir')
os.makedirs(tmp_dir, exist_ok=True)

video_file = "%s"%os.path.join(tmp_dir, 'video.avi')
audio_file = os.path.join(tmp_dir, 'audio.wav')

os.makedirs(tmp_dir, exist_ok=True)

input_vid = "%s"%input_vid
video_file = "%s"%os.path.join(tmp_dir, 'video.avi')
audio_file = os.path.join(tmp_dir, 'audio.wav')

try:        
    cap = cv2.VideoCapture(input_vid)

    start_end_table = np.genfromtxt(timestamp_file)
    start_end_table = start_end_table[1:,:]
    output_file     = os.path.join(os.path.split(timestamp_file)[0], os.path.splitext(os.path.basename(timestamp_file))[0] + '.avi')
    
    for times in start_end_table:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vOut   = cv2.VideoWriter(video_file, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4))))
        
        start_ms = times[0]*1000
        end_ms   = times[1]*1000
        cap.set(0, start_ms)
        
        while (True):
            ret, frame = cap.read()
            
            if ((ret == True) and (cap.get(0) <= end_ms)):
                vOut.write(frame)
                continue
                
            else:
                vOut.release()
                
                audio_start, audio_end = times[0], times[1]
                
                file = os.path.join(tmp_dir, 'video_' + ('000' + str(k))[-3:] + '.avi')
                k +=1
    
                command = ("ffmpeg -y -i \"%s\" -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f \"%s\"" \
                           % (input_vid, audio_start, audio_end, audio_file)) #-async 1
                output = subprocess.call(command, shell=True, stdout=None)
                
                command = ("ffmpeg -y -i \"%s\" -i \"%s\" -c:v copy -c:a copy \"%s\"" % (video_file, audio_file, file)) #-async 1 
                output = subprocess.call(command, shell=True, stdout=None)
                
                subprocess.call("rm \"%s\""%(video_file), shell=True)
                subprocess.call("rm \"%s\""%(audio_file), shell=True)
                break
                
    cap.release()    
except:
    vOut.release()
    cap.release()    


# Concatenate all vids
videos = np.sort(glob.glob(tmp_dir + '/video_***.avi'))
videos = np.array(['file ' + os.path.abspath(k).replace(" ", "\ ") for k in videos])

video_files_dir  = os.path.join(tmp_dir, 'video_files.txt')
concatenated_vid = os.path.join(output_file)

np.savetxt(video_files_dir, videos, fmt='%s')

video_files_dir_  = '"%s"'%video_files_dir
concatenated_vid_ = '"%s"'%concatenated_vid

command = 'ffmpeg -y -f concat -safe 0 -i %s -codec copy %s'%(video_files_dir_, concatenated_vid_)
output = subprocess.call(command, shell=True, stdout=None)

command = 'rm -r %s'%(tmp_dir)
output = subprocess.call(command, shell=True, stdout=None)
