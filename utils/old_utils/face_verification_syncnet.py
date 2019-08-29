#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:30:04 2019

@author: dcaulley
"""

import pickle
import cv2

import os, sys
import subprocess

import numpy as np
import scipy.spatial.distance as dist

from keras_vggface.vggface import VGGFace

exp_dir            = sys.argv[1]
curr_vid           = sys.argv[2]
template_file      = sys.argv[3]
scoring_type       = sys.argv[4]
threshold          = float(sys.argv[5])

pywork_dir      = os.path.join(exp_dir, 'pywork', curr_vid)
pycrop_dir      = os.path.join(exp_dir, 'pycrop', curr_vid)
pycrop_dir      = '"%s"'%pycrop_dir


faces_file      = os.path.join(pywork_dir,     'faces.pckl')
tracks_file      = os.path.join(pywork_dir,    'tracks.pckl')
alltracks_file  = os.path.join(pywork_dir, 'alltracks.pckl')


template_faces = np.load(template_file)

with open(tracks_file, 'rb') as fil:
    track = pickle.load(fil)
    
with open(faces_file, 'rb') as fil:
    faces = pickle.load(fil)
    
with open(alltracks_file, 'rb') as fil:
    alltracks = pickle.load(fil)


resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')




def face_verif(features):    
    scores = []
    u = features
    for v in template_faces.T:
        if scoring_type == 'cosine':
            scores.append(dist.cosine(u, v))
        else:
            scores.append(dist.euclidean(u, v)) 
    return np.mean(scores)

def img_reformat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return img 

#cap = cv2.VideoCapture('/Users/dcaulley/projects/audio_visual_diarization/research_exp/pyavi/out_out/video.avi')

frame_num = 1
all_frames = []
all_rects  = []
for tr in alltracks:
    frames_ = tr[0]
    rects_  = tr[1]
    
    #pts = np.random.randint(len(frames_),size=10)

    pts = np.arange(len(frames_))
    np.random.shuffle(pts)
    pts = pts[:20]
    pts.sort()
    
    frames_ = frames_[pts]
    rects_  = rects_[pts]
    
    all_frames.append(frames_)
    all_rects.append(rects_)
all_frames = np.array(all_frames)


frame_num = 0


cap = cv2.VideoCapture(os.path.join(exp_dir, 'pyavi', curr_vid, 'video.avi'))
fw, fh = cap.get(3), cap.get(4)

face_scores = np.zeros(all_frames.shape, dtype=float)
while True:
    ret, frame = cap.read()
    rows_, cols_ = np.where(all_frames == frame_num)
    
    for r, c in zip(rows_, cols_):
        y_s, x_s, y_e, x_e = all_rects[r][c,:]
        y_s, x_s, y_e, x_e = int(y_s*fh), int(x_s*fw), int(y_e*fh), int(x_e*fw)
        face = frame[y_s:y_e, x_s:x_e, :]
        face = img_reformat(face)
        features = resnet50_features.predict(face)
        
        face_scores[r,c] = face_verif(features)
    
    frame_num +=1
    
    if not ret:
        break
    
face_scores = np.mean(face_scores, 1)

new_tracks = []
d = 0
for k, score in enumerate(face_scores):
    if score > threshold:
        command = ('rm -f %s/%s.avi' % (pycrop_dir,  ('000000' + str(k))[-5:])   )
        subprocess.check_call(command, shell=True, stdout=None)
    else:
        command = ('mv %s/%s.avi %s/%s.avi' % (pycrop_dir, ('000000'+str(k))[-5:], pycrop_dir, ('000000'+str(d))[-5:]) )
        subprocess.check_call(command, shell=True, stdout=None)
        new_tracks.append(track[k])
        d = d + 1

        
with open(tracks_file, 'wb') as fil:
    track = pickle.dump(new_tracks, fil)