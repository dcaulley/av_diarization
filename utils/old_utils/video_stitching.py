#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:21:47 2019

@author: dcaulley: Desmond Caulley
"""

import glob
import os, sys
import subprocess

import cv2

import numpy as np

face_dir        = sys.argv[1]
input_vid       = sys.argv[2]
cropped_frames  = int(sys.argv[3])
output_vids_dir = sys.argv[4]
concatenate_vid = int(sys.argv[5])
fps             = int(sys.argv[6])



verified_faces  = np.load(face_dir)
verified_frames = verified_faces[0,:]


#verified_faces_smoothing
frame_num = 0
faces_detected = []

cap = cv2.VideoCapture(input_vid)
cap.set(cv2.CAP_PROP_FPS, fps)

orig_width, orig_height = int(cap.get(3)), int(cap.get(4))
ratio = (orig_width/500.0)
pad = 20



start_time = verified_frames[0]
start_end_table = []
for f in range(len(verified_frames)):
    try:
        if verified_frames[f+1] > verified_frames[f] + 13:
            end_time   = verified_frames[f]
            
            if end_time - start_time >= 13:
                start_end_table.append([start_time, end_time])
            start_time = verified_frames[f+1]
    except:
        continue
coord_pad= 0

# Still need to do some work here since it might squish the entire
if cropped_frames:
    for i, seg in enumerate(start_end_table):
        start_ind = np.where(verified_frames == seg[0])[0][0]
        end_ind   = np.where(verified_frames == seg[1])[0][0]
        
        #check out medfilt for filtering in the futures
        curr_faces = verified_faces[:, start_ind:end_ind+1]
        
        left   = min(curr_faces[1,:])
        right  = (max(curr_faces[1,:]) + max(curr_faces[3,:]))
        
        top    = min(curr_faces[2,:])
        bottom = (max(curr_faces[2,:]) + max(curr_faces[4,:]))
        
        w = right - left
        h = bottom - top
        
        if float(max(w,h))/min(w,h) < 1.5:
            curr_faces[1,:] = max(0, min(curr_faces[1,:]) - coord_pad)
            curr_faces[2,:] = max(0, min(curr_faces[2,:]) - coord_pad)
            curr_faces[3,:] = w + 2*coord_pad
            curr_faces[4,:] = h + 2*coord_pad
        else:
            curr_faces[1,:] = 0
            curr_faces[2,:] = 0
            curr_faces[3,:] = orig_width
            curr_faces[4,:] = orig_height

        

fourcc = cv2.VideoWriter_fourcc(*'XVID')

for i, seg in enumerate(start_end_table):
    
    video_file = "%s"%os.path.join(output_vids_dir, 'video.avi')
    
    if cropped_frames:
        vOut = cv2.VideoWriter(video_file, fourcc, cap.get(5), (224,224))
    else:
        vOut = cv2.VideoWriter(video_file, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4))))

    cap.set(cv2.CAP_PROP_POS_FRAMES,seg[0])
    frame_num = seg[0]
    
    while (True):
        ret, frame = cap.read()
        
        if ((ret == True) and (frame_num <= seg[1])):
            if frame_num in verified_frames:
                indx = np.where(verified_faces == frame_num)[1][0]
                x, y, w, h = np.rint(ratio*(verified_faces[:,indx][1:])).astype(int)
                
                if cropped_frames:
                    face = frame[max(0, y - pad):y+h+pad, max(x, x - pad):x+w+pad]
                    vOut.write(cv2.resize(face,(224,224)))
                else:
                    vOut.write(frame)
            else:
                if cropped_frames:
                    vOut.write(cv2.resize(face,(224,224)))
                else:
                    vOut.write(frame)
            
            frame_num +=1
        else:
            vOut.release()
            
            audio_start = seg[0]*(1/fps)
            audio_end   = seg[1]*(1/fps)
            
            audio_file = os.path.join(output_vids_dir, 'audio.wav')
            file = os.path.join(output_vids_dir, 'video_' + ('000' + str(i))[-3:] + '.avi')

            command = ("ffmpeg -y -i \"%s\" -ac 1 -vn -acodec pcm_s16le -ar 16000 -ss %.3f -to %.3f \"%s\"" \
                       % (input_vid, audio_start, audio_end, audio_file)) #-async 1
            output = subprocess.call(command, shell=True, stdout=None)
            
            command = ("ffmpeg -y -i \"%s\" -i \"%s\" -c:v copy -c:a copy \"%s\"" % (video_file, audio_file, file)) #-async 1 
            output = subprocess.call(command, shell=True, stdout=None)


            subprocess.call("rm \"%s\""%(video_file), shell=True)
            subprocess.call("rm \"%s\""%(audio_file), shell=True)

            break

if concatenate_vid:
    videos = np.sort(glob.glob(output_vids_dir + '/video_***.avi'))
    videos = np.array(['file ' + os.path.abspath(k).replace(" ", "\ ") for k in videos])
    
    video_files_dir  = os.path.join(output_vids_dir, 'video_files.txt')
    concatenated_vid = os.path.join(output_vids_dir, 'videos_concat.avi')
    
    np.savetxt(video_files_dir, videos, fmt='%s')

    video_files_dir_  = '"%s"'%video_files_dir
    concatenated_vid_ = '"%s"'%concatenated_vid
    
    print(video_files_dir_)
    print(concatenated_vid_)
    command = 'ffmpeg -y -f concat -safe 0 -i %s -codec copy %s'%(video_files_dir_, concatenated_vid_)
    output = subprocess.call(command, shell=True, stdout=None)
    print("Desmond")


        
np.save(os.path.join(output_vids_dir, 'frame_times.npy'), start_end_table)

cap.release()