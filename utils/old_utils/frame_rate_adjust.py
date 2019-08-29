#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:22:03 2019

@author: dcaulley

This script has bug. In case fps_desired > fps_orig, upsampling doesn't work.
"""


import os, sys, subprocess

input_video  = sys.argv[1]
frame_rate   = sys.argv[2]
output_video = sys.argv[3]

input_video  = '"%s"'%input_video
output_video = '"%s"'%output_video

print('\nStarted Frame Rate Conversion\n')

#command = ("ffmpeg -y -i %s -r %s -profile:v baseline -c:a copy -strict -2 %s" % (input_video, frame_rate, output_video))
#command = ("ffmpeg -y -i %s -r %s -profile:v baseline -c:a copy -strict -2 %s" % (input_video, frame_rate, output_video))

command = ("ffmpeg -y -i %s -async 1 -r %s -deinterlace -strict -2 %s" % (input_video, frame_rate, output_video))

FNULL=open(os.devnull,'w')
output = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)