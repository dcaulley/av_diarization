#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:20:21 2019

@author: dcaulley
"""

import os, sys
import subprocess

import glob
import pickle

import numpy as np
from scipy import signal

frame_times  = sys.argv[1]
video_dir    = sys.argv[2]
syncnet_dir  = sys.argv[3]
exp_dir      = sys.argv[4]
output_file  = sys.argv[5]
fps          = int(sys.argv[6])
video_concat = int(sys.argv[7])

frame_dur   = 1/float(fps)
frame_times = np.load(frame_times)

exp_dir_     ='"%s"'%exp_dir

def audio_lip_tracks(tracks_file, actives_file):    
    with open(tracks_file, 'rb') as fil:
        tracks = pickle.load(fil)
    
    with open(actives_file, 'rb') as fil:
        dists = pickle.load(fil)
        
    
    faces = [ [] for i in range(1000000) ]
    k=0
    
    for ii, track in enumerate(tracks):
    
    	mean_dists =  np.mean(np.stack(dists[ii],1),1)
    	minidx = np.argmin(mean_dists,0)
    	#minval = mean_dists[minidx] 
    	
    	fdist   	= np.stack([dist[minidx] for dist in dists[ii]])
    	fdist   	= np.pad(fdist, (2,4), 'constant', constant_values=10)
    	fdist_mf	= signal.medfilt(fdist,kernel_size=19)
    
    	for ij, frame in enumerate(track[0][0].tolist()) :
            try:
                faces[frame].append([ii, fdist_mf[ij], track[1][0][ij], track[1][1][ij], track[1][2][ij]])
                k = k + 1
            except:
                continue
            
    faces = faces[:k]
    return faces

def running_syncnet(vid):
    vid = '"%s"'% vid
    
    run_pipeline = "\"./" + os.path.join(syncnet_dir, 'run_pipeline.py') + "\""
    
    command_a = ('CUDA_VISIBLE_DEVICES=7 python %s --videofile %s --reference %s --data_dir %s' \
                 % (run_pipeline, vid, 'sync_out', exp_dir_))

    subprocess.call(command_a, shell=True, stdout=None)

    try:
        run_syncnet  = "\"./" + os.path.join(syncnet_dir, 'run_syncnet_cuda.py')  + "\""
        command_b = ('CUDA_VISIBLE_DEVICES=7 python %s --videofile %s --reference %s --data_dir %s' \
                 % (run_syncnet,  vid, 'sync_out', exp_dir_))
        subprocess.check_call(command_b, shell=True, stdout=None)
        
    except:
        run_syncnet  = "\"./" + os.path.join(syncnet_dir, 'run_syncnet.py')  + "\""
        command_b = ('CUDA_VISIBLE_DEVICES=7 python %s --videofile %s --reference %s --data_dir %s' \
                 % (run_syncnet,  vid, 'sync_out', exp_dir_))
        subprocess.check_call(command_b, shell=True, stdout=None)

    
    tracks_file  = os.path.join(exp_dir, 'pywork/sync_out/tracks.pckl')
    actives_file = os.path.join(exp_dir, 'pywork/sync_out/activesd.pckl')
    
    return tracks_file, actives_file


audio_visual_match = []
if (not video_concat):
    videos = np.sort(glob.glob(video_dir + '/video_***.avi'))
        
    for (segs, vid) in zip(frame_times, videos):
        
        tracks_file, actives_file = running_syncnet(vid)
        
        audio_lip_tr = audio_lip_tracks(tracks_file, actives_file)
            
        for f in np.arange(segs[1] - segs[0]):
            try:
                if audio_lip_tr[f] != []:
                    audio_visual_match.append(f+segs[0])
            except:
                continue
        
        print("DONE WITH VIDEO: ", vid)

else:
    vid = np.sort(glob.glob(video_dir + '/videos_concat.avi'))[0]
    tracks_file, actives_file = running_syncnet(vid)
    audio_lip_tr = audio_lip_tracks(tracks_file, actives_file)
    
    all_frames = []
    [all_frames.extend(np.arange(a[0], a[1])) for a in frame_times]
    
    for f in range(len(all_frames)):
        try:
            if audio_lip_tr[f] != []:
                audio_visual_match.append(all_frames[f])
        except:
            continue
    
    
start_end_table =[];
start_time = round(audio_visual_match[0]*frame_dur, 2)
for f in range(len(audio_visual_match)):
    try:
        if audio_visual_match[f+1] > audio_visual_match[f] + 4:
            end_time = round(audio_visual_match[f]*frame_dur, 2)      
            start_end_table.append([start_time, end_time])
            start_time = round(audio_visual_match[f+1]*frame_dur, 2)
    except:
        continue

start_end_table.append([start_time, audio_visual_match[f]*frame_dur])
start_end_table = np.array(start_end_table).astype(str)

start_end_table = np.insert(start_end_table, 0, ['start', 'end'], axis=0)
np.savetxt(output_file, start_end_table, fmt='%s')
    