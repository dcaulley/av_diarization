#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:52:38 2019

@author: dcaulley: Desmond Caulley

This script takes as input the sync-net algorithm outputs. It uses the results
to come up with time intervals when face/lips and audio are synchronized
"""

import os
import pickle
import argparse

import numpy as np


# Parsing Arguments
parser = argparse.ArgumentParser(description = "processes output of sync-net and saves times when lip-syncs with audio")

parser.add_argument('--exp_dir',      type=str,    default='',   help='experiment directory with videos')
parser.add_argument('--video_id',     type=str,    default='',   help='id of video of interest')
parser.add_argument('--output_file',  type=str,    default='',   help='output file to save times')
parser.add_argument('--track_offset', type=int,    default=9,    help='if track offset > track_offset, ignore track')
parser.add_argument('--frame_conf',   type=float,  default=1,    help='if frame confidence of match > frame_conf, keep frame')
parser.add_argument('--min_time',     type=float,  default=1,    help='if identified segment > min_time, keep that segment')



opt = parser.parse_args();

exp_dir      = opt.exp_dir
curr_vid     = opt.video_id
output_file  = opt.output_file
track_offset = opt.track_offset
frame_conf   = opt.frame_conf
min_time     = opt.min_time






pywork_dir       = os.path.join(exp_dir, curr_vid, 'pywork')

tracks_file      = os.path.join(pywork_dir, 'tracks.pckl')
activesd_file    = os.path.join(pywork_dir, 'activesd.pckl')
frame_confs_file = os.path.join(pywork_dir, 'frame_confs.pckl')
offsets_file     = os.path.join(pywork_dir, 'offsets.txt')


with open(tracks_file, 'rb') as fil:
    tracks = pickle.load(fil)    

with open(activesd_file, 'rb') as fil:
    actives_d = pickle.load(fil)
    
with open(frame_confs_file, 'rb') as fil:
    frame_confs = pickle.load(fil)


def audio_lip_tracks(tracks, dists, frame_confs, offs, confs):
    faces = [ [] for i in range(1000000) ]
    
    for ii, track in enumerate(tracks):
        if abs(offs[ii]) > track_offset:
            continue
        
        for ij, frame in enumerate(track[0][0].tolist()) :
            try:
                if frame_confs[ii][ij] > frame_conf:
                    faces[frame].append([ii, frame_confs[ii][ij]])
            except:
                continue
    try:
        faces = faces[:frame]
    except:
        pass
    return faces


offsets = np.loadtxt(offsets_file, dtype=str)
confs   = offsets[1:,2].astype(float)
offs    = offsets[1:,1].astype(float)
faces   = audio_lip_tracks(tracks, actives_d, frame_confs, offs, confs)

empty_space_list_start = []
empty_space_list_end   = []

sp = 0
for k, f in enumerate(faces):
    if f == []:
        sp += 1
        if sp == 3:
            empty_space_list_end.append(k-2)
    elif f != [] and sp >= 3:
        empty_space_list_start.append(k-1)
        sp = 0
        
empty_space_list_end   = empty_space_list_end[1:]
empty_space_list_end   = empty_space_list_end[:len(empty_space_list_start)]
for st,end in zip(empty_space_list_start, empty_space_list_end):
    try:
        if (end - st) < 13:
            faces[st:end] = [[]]*(end - st)
    except:
        continue
            

audio_visual_match = []
for k, idx in enumerate(faces):
    if idx != []:
        audio_visual_match.append(k)

frame_dur=1/25.0
start_end_table =[];
start_time = round(audio_visual_match[0]*frame_dur, 2)
      
for f in range(len(audio_visual_match)):
    try:
        if audio_visual_match[f+1] > audio_visual_match[f] + 12:
            end_time = round(audio_visual_match[f]*frame_dur, 2)  
            
            if end_time - start_time >= min_time:
                start_end_table.append([start_time, end_time])
            start_time = round(audio_visual_match[f+1]*frame_dur, 2)
    except:
        continue

if (audio_visual_match[f]*frame_dur - start_time) > min_time:
    start_end_table.append([start_time, audio_visual_match[f]*frame_dur])

if (len(start_end_table) > 0):
    start_end_table = np.array(start_end_table).astype(str)
    
    start_end_table = np.insert(start_end_table, 0, ['start', 'end'], axis=0)
    np.savetxt(output_file, start_end_table, fmt='%s')

