#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:42:42 2019

@author: dcaulley: Desmond Caulley

This script takes as input a search_term which could be a phrase and uses youtube-dl to 
download the top N results. The num of videos to download, N, is given by num_videos.
"""

import os
import argparse

import urllib.request
from bs4 import BeautifulSoup



parser = argparse.ArgumentParser(description = "downloads videos from youtube")

parser.add_argument('--search_term', type=str,   default='Elie Khoury interview', help='eg. \"Elie Khoury interview\"')
parser.add_argument('--num_videos',  type=int,   default=20,                      help='num_videos to download')
parser.add_argument('--output_dir',  type=str,   default='youtube_vids',          help='dir to store videos')
parser.add_argument('--fps',         type=str,   default=25,                      help='fps to select if available')


opt = parser.parse_args();



def youtube_top_results(textToSearch):
    query = urllib.parse.quote(textToSearch)
    url = "https://www.youtube.com/results?search_query=" + query
    response = urllib.request.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
    video_list = []
    for vid in soup.findAll(attrs={'class':'yt-uix-tile-link'}):
        video_list.append('https://www.youtube.com' + vid['href'])
    return video_list
        



os.makedirs(opt.output_dir, exist_ok = True)    
search_term = '"%s"'%opt.search_term

try:
    os.chdir(opt.output_dir)
    os.system("youtube-dl --id -f 'bestvideo[ext!=webm][fps=%s]+bestaudio[ext!=webm]' -f mp4 --ignore-errors ytsearch%s:%s"%(opt.fps, opt.num_videos, search_term))
    #os.system("youtube-dl --id -f 'bestvideo[fps=25]+bestaudio' --ignore-errors ytsearch%s:%s"%(num_videos, search_term))

except:
    pass


print('\nDone with Downloading Youtube Videos\n')
