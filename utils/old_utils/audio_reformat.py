#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:22:03 2019

@author: dcaulley
"""


import sys
import ffmpeg

input_video  = sys.argv[1]
frame_rate   = sys.argv[2]
output_video = sys.argv[3]

ffmpeg.input(input_video).filter('fps', fps=frame_rate, 
            round='up').output(output_video).run()