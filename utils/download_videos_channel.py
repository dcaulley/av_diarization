#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:28 2019

@author: dcaulley: Desmond Caulley

This script downloads video from a given youtube_channel. The channel is filtered using
text_matching with the title. User can select number of videos to download
"""

import os
import sys
import argparse


import urllib.request
from bs4 import BeautifulSoup


parser = argparse.ArgumentParser(description = "downloads videos from a youtube channel")

parser.add_argument('--youtube_channel',  type=str,   default='',              help='url for youtube channel')
parser.add_argument('--match_term',       type=str,   default='interview',     help='eg. interview')
parser.add_argument('--num_videos',       type=int,   default=30,              help='num videos to download')
parser.add_argument('--output_dir',       type=str,   default='channel_vids',  help='dir to store videos')
parser.add_argument('--fps',              type=int,   default=25,              help='fps to select if available')
#youtube_channel -- eg. "https://www.youtube.com/user/nessmatvreplays"

opt = parser.parse_args();



os.makedirs(opt.output_dir, exist_ok = True) 
search_term = '"%s"'%opt.youtube_channel
try:
    os.chdir(opt.output_dir)
    os.system("youtube-dl --id -f 'bestvideo[ext!=webm][fps=%s]+bestaudio[ext!=webm]' -f mp4 \
        --ignore-errors --match-filter \"duration > 600\"  --max-downloads %s --match-title %s %s"%(opt.fps, opt.num_videos, opt.match_term, opt.youtube_channel))

except Exception as e:
	print(e)
	pass

print('\nDone with Downloading Youtube Videos\n')
