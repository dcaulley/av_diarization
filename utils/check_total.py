#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:03:45 2019

@author: dcaulley: Desmond Caulley

This script takes in as argument directory with a list of text files with time stamps.
The script goes through the time stamps and adds them all up and returns total time.
"""

import sys, glob
import numpy as np

folder = sys.argv[1]

files = glob.glob(folder + "/*.txt")

total_time = []
for file in files:
    try:
        info = np.genfromtxt(file)
        times = info[1:,:]
        total_time.append(sum(times[:,1] - times[:,0]))
    except Exception as e:
    	print(e)
    	pass

print(int(sum(total_time)))
    