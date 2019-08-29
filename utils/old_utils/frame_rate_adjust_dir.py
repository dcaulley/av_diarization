#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:19:57 2019

@author: dcaulley: Desmond Caulley

Converts an entire directory of video from arbitrary frame rate to 25 fps
This script has a bug. In case where fps is less than 25 fps, you can't upsample
back to 25 fps.
"""

import   os, sys, subprocess
import   glob

folder     = sys.argv[1]
frame_rate = sys.argv[2]

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


frame_rate_adjust(folder, frame_rate)
        