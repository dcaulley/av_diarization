#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:42:23 2019

@author: dcaulley: Desmond Caulley
"""

"""
This script goes through entire text files with start and end times in a 
directory and tries to stitched video files using those time stamps.
Each txt file with time-stamp comes from a unique video.

The script takes in directory with timestamp. And output file to store
concatenated video 
Eg ./video_stitching_all time_stamp_dir/ output_filename.avi
"""


import glob
import os, sys
import subprocess

import cv2
import numpy as np


exp_dir     = sys.argv[1] #directory with .txt files with timestamps
output_file = sys.argv[2] #output_video filename. eg. video.avi


files   = glob.glob(exp_dir + '/*.txt')
tmp_dir = os.path.join(exp_dir, 'tmp_dir')
os.makedirs(tmp_dir, exist_ok=True)

video_file = "%s"%os.path.join(tmp_dir, 'video.avi')
audio_file = "%s"%os.path.join(tmp_dir, 'audio.wav')


k = 0
for file in files:
    try:
        base_name = os.path.basename(file)
        input_vid = "%s"%os.path.join(exp_dir, 'video_processing', \
                                       os.path.splitext(base_name)[0], 'pyavi', 'video.avi')
        cap = cv2.VideoCapture(input_vid)

        start_end_table = np.genfromtxt(file)
        start_end_table = start_end_table[1:,:]
        
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
                    
                    file = "%s" % os.path.join(tmp_dir, 'video_' + ('000' + str(k))[-3:] + '.avi')
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
    except Exception as e:
        print(e)
        cap.release()    
        continue
    
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

command = 'rm -rf %s'%('"%s"'%tmp_dir)
output = subprocess.call(command, shell=True, stdout=None)
    