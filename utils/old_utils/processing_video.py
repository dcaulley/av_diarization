#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:45:21 2019

@author: dcaulley
"""

import   os, sys, subprocess
import   glob


def frame_rate_adjust(folder, frame_rate=25):
    
    files = glob.glob(folder + '/*')
    for file in files:
        
        input_video  = "\"" + file + "\""
        frame_rate   = frame_rate
        output_video = "\"" + folder + "/video.mp4" + "\""
        
        
        print('Started Frame Rate Conversion')
        
        command = ("ffmpeg -y -i %s -r %s -profile:v baseline -c:a copy -strict -2 %s" % (input_video, frame_rate, output_video))
        FNULL=open(os.devnull,'w')
        subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        
        command2 = ("mv %s %s" % (output_video, input_video))
        subprocess.call(command2, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)


all_videos = sys.argv[1]
frame_rate = 25

folders = glob.glob(all_videos + '/*')[6:]
for f in folders:
    frame_rate_adjust(f, frame_rate)

