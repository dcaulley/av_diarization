#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:24:41 2019

@author: dcaulley
"""

import os, cv2
import sys, subprocess

videofile = sys.argv[1]
outputfile = sys.argv[2]

cap = cv2.VideoCapture(videofile)
fps = round(cap.get(5))
if cap.get(5) >= 25:
	fps = 25

os.makedirs(os.path.split(outputfile)[0], exist_ok=True)

command = "echo %d > %s"%(fps, '"%s"'%outputfile)
output  = subprocess.call(command, shell=True, stderr=subprocess.STDOUT)