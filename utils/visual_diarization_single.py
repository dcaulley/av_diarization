#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:15:28 2019

@author: dcaulley: Desmond Caulley
"""


import pickle
import cv2
import os, sys

import argparse
import numpy as np

import scipy.spatial.distance as dist

import tensorflow as tf
from keras import backend as K
from keras_vggface.vggface import VGGFace
from sklearn.cluster import AgglomerativeClustering, DBSCAN


parser = argparse.ArgumentParser(description = "video diarization in single video")

parser.add_argument('--exp_dir',    type=str,   default='',         help='exp_dir where videos are located.'  )
parser.add_argument('--curr_vid',   type=str,   default='',         help='video ID')
parser.add_argument('--output_dir', type=str,   default='',         help='output directory')
parser.add_argument('--track_only', type=int,   default=1,          help='1 if you want to ignore nontrack frames')
parser.add_argument('--min_time',   type=int,   default=0,          help='if person appears greater than this value, we keep person times'      )
parser.add_argument('--save_feats', type=int,   default=1,          help='save feature vector representing a person')
parser.add_argument('--num_feats',  type=int,   default=50,         help='num features used to sample frames in a track')
parser.add_argument('--save_person_faces',  type=int,   default=1,  help='save faces for each person')

opt = parser.parse_args();

exp_dir           = opt.exp_dir
curr_vid          = opt.curr_vid
output_dir        = opt.output_dir
track_only        = opt.track_only
min_time          = opt.min_time
save_feats        = opt.save_feats
num_feats         = opt.num_feats
save_person_faces = opt.save_person_faces 


scoring_type   = "cosine"
threshold      = 0.4


track_only, min_time  = int(track_only), int(min_time)
save_feats, num_feats = int(save_feats), int(num_feats)

os.makedirs(output_dir, exist_ok=True)


pywork_dir      = os.path.join(exp_dir, curr_vid, 'pywork')
pycrop_dir      = os.path.join(exp_dir, curr_vid, 'pycrop')
pycrop_dir      = '"%s"'%pycrop_dir

faces_file      = os.path.join(pywork_dir,     'faces.pckl')
tracks_file     = os.path.join(pywork_dir,    'tracks.pckl')
alltracks_file  = os.path.join(pywork_dir, 'alltracks.pckl')



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.22
session = tf.Session(config=config)
K.set_session(session)



with open(tracks_file, 'rb') as fil:
    track = pickle.load(fil)
    
with open(faces_file, 'rb') as fil:
    faces = pickle.load(fil)
    
with open(alltracks_file, 'rb') as fil:
    alltracks = pickle.load(fil)


resnet50_feats = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')




def img_reformat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return img



def face_verif(feats, tr_feats):    
    scores = []
    u = feats
    for v in tr_feats:
        if scoring_type == 'cosine':
            scores.append(dist.cosine(u, v))
        else:
            scores.append(dist.euclidean(u, v)) 
    return scores


def resort_pred(pred_arr):
    pred_arr      = np.array(pred_arr)
    labels_, ind_ = np.unique(pred_arr, return_index=True)
    #labels_ = labels_[ind_]
    final_pred  = np.ones((len(pred_arr), ))*-2
    for k, lab in enumerate(labels_):
        final_pred[np.where(pred_arr == lab)] = k
    return final_pred



def track_frames_faces_select(num, all_tracks):
    tr_fr_all     = []
    tr_fr_mini    = []
    tr_faces_mini = []
    for tr in all_tracks:
        frames_, rects_ = tr[0], tr[1]
        
        rects_mod = np.ones((num,4))
        tr_fr_all.append(frames_)
            
        pts = np.arange(len(frames_))
        np.random.shuffle(pts)
        pts = pts[5:num]
        pts.sort()
        
        frames_, rects_ = frames_[pts], rects_[pts]
        
        frames_ = np.concatenate((frames_, np.ones((max(0,num-len(frames_)),))*-1))
        rects_mod[:len(rects_), :] = rects_
        
        tr_fr_mini.append(frames_)
        tr_faces_mini.append(rects_mod)
    tr_fr_mini = np.array(tr_fr_mini)
    return tr_fr_all, tr_faces_mini, tr_fr_mini



def obtain_track_nontrack_features(tr_frames_mini, tr_faces_mini):
    len_faces = len(faces)

    cap = cv2.VideoCapture(os.path.join(exp_dir, curr_vid, \
        'pyavi', 'video.avi'))
    fw, fh = cap.get(3), cap.get(4)

    non_tr_feats = []
    tr_feats = [ [] for i in range(10000) ]

    tr_labels, non_tr_frames = [], []

    while True:
        ret, frame = cap.read()
        
        if (not ret) or (len_faces <= int(cap.get(1))):
            break
        
        rows_, cols_ = np.where(tr_frames_mini == cap.get(1))
        
        if len(rows_) != 0:
            for r, c in zip(rows_, cols_):
                y_s, x_s, y_e, x_e = tr_faces_mini[r][c,:]
                y_s, x_s, y_e, x_e = int(y_s*fh), int(x_s*fw), int(y_e*fh), int(x_e*fw)
                face = frame[y_s:y_e, x_s:x_e, :]
                face = img_reformat(face)
                feats = resnet50_feats.predict(face)
                tr_labels.append(r)
                tr_feats[r].append(feats)
        else: 
            if not track_only:
                if faces[int(cap.get(1))] != []:
                    for rect in faces[int(cap.get(1))]:
                        y_s, x_s, y_e, x_e = rect[1]
                        y_s, x_s, y_e, x_e = int(y_s*fh), int(x_s*fw), int(y_e*fh), int(x_e*fw)
                        face = frame[y_s:y_e, x_s:x_e, :]
                        face = img_reformat(face)
                        feats = resnet50_feats.predict(face)
                        non_tr_feats.append(feats)
                        non_tr_frames.append(int(cap.get(1)))
                    
    return tr_labels, non_tr_frames, tr_feats, non_tr_feats







# Select random frames from tracks to do clustering with
track_frames_all, track_faces_mini, track_frames_mini = \
                            track_frames_faces_select(num_feats, alltracks)





# Extract features for the selected frames                    
track_labels, non_track_frames, track_feats, non_track_feats = \
            obtain_track_nontrack_features(track_frames_mini, track_faces_mini)

track_feats = track_feats[:len(track)]
track_feats = [np.concatenate(k) for k in track_feats]





# Agglomerative clustering of track frames
input_feats  = np.array([np.mean(k, axis=0) for k in track_feats])
track_labels = np.arange(len(input_feats))

agg_cluster  = AgglomerativeClustering(n_clusters=None, affinity='cosine', \
                                       linkage='average', distance_threshold=0.35)

track_preds_     = agg_cluster.fit_predict(input_feats).astype(int)
track_preds      = resort_pred(track_preds_).astype(int) 

track_pred_feats = input_feats.copy()





# Grouping feats with same label together:
track_pred_feats_= {}
for k in np.unique(track_preds):
    pp_feats = track_pred_feats[np.where(track_preds == k)[0]]
    
    pp_feats = np.mean(pp_feats, axis = 0)
    
    track_pred_feats_.update({k: pp_feats})




# Comparing non_track face to each track to see if it belong with one of tracks
if not track_only:
    non_track_scores = []
    non_track_feats = np.concatenate(non_track_feats)
    track_feats_avg = np.array([np.mean(feats, axis=0) for feats in track_feats])
    
    for feat in non_track_feats:
        curr_scores = face_verif(feat, track_feats_avg)
        non_track_scores.append(curr_scores)
    
    non_track_scores  = np.array(non_track_scores)
    non_track_scores_ = non_track_scores.copy()
    non_track_pred    = np.argmin(non_track_scores, axis=1)
    non_track_scores  = np.min(non_track_scores, axis=1)



# DB clustering of non_track faces.
# Non_track scores which didn't match tracks ( > 0.45) will have own cluster
if not track_only:
    dbscan_cluster = DBSCAN(metric="cosine", eps=0.1)
    non_track_dbcluster_preds = dbscan_cluster.fit_predict(non_track_feats) \
                                        + (max(track_preds) + 2)
    
    non_track_pred = track_preds[non_track_pred]
    non_track_pred[np.where(non_track_scores >= 0.4)[0]] = \
                    non_track_dbcluster_preds[np.where(non_track_scores >= 0.4)[0]]



# This scripts makes final predictions on faces
# across track and non_track frames 
len_faces  = len(faces)
face_preds = [ [] for i in range(len_faces) ]

for k, tr in enumerate(track_frames_all):
    for fr in tr:
        face_preds[fr].append(int(track_preds[k]))
        
if not track_only:       
    for k, fr in enumerate(non_track_frames):
        face_preds[fr].append(int(non_track_pred[k]))
    
for k in range(len(face_preds)):
    face_preds[k] = list(set(face_preds[k]))
    face_preds[k].sort()
    


# Puts frames for each individual in its own track
unique_labels = np.unique(np.concatenate(face_preds)).astype(int)
av_output_fr  = []
for l in unique_labels:
    person_fr =[]
    for k in range(len_faces):
        if l in face_preds[k]:
            person_fr.append(k)
    av_output_fr.append([person_fr, l])

output_person_faces=[]
for k in unique_labels:
    loc = np.where(track_preds == k)[0]
    track_info = np.array(alltracks)[loc]
    track_frame_loc = np.concatenate(([k[0] for k in track_info]))
    track_frame_faces = np.concatenate(([k[1] for k in track_info]))
    output_person_faces.append([track_frame_loc, track_frame_faces])

# Smoothing of frame times when individual appears
final_times = []
feats_to_save  = []
for pr, label in av_output_fr:
    person_f   = []
    total_time = 0
    start = pr[0]
    for k in range(1, len(pr)):
        if pr[k] > pr[k-1] + 7:
            end = pr[k-1]
            if (end - start >= 7):
                ss, ee = start/25., end/25.
                person_f.append([ss, ee])
                total_time += (ee - ss)
            start = pr[k]
    end = pr[-1]
    if (end - start >= 7):
        ss, ee = start/25., end/25.
        person_f.append([ss, ee])
        total_time += (ee - ss)
    
    if total_time > min_time:
        final_times.append([person_f, label])
        
        if track_only and save_feats:
            feats_to_save.append([track_pred_feats_[label], label])
            
        



# Writing appearances times for each individual
os.makedirs(output_dir, exist_ok=True)

for se, label in final_times:
    se = np.array(se).astype(str)
    start_end_table = np.insert(se, 0, ['start', 'end'], axis=0)
    output_file     = os.path.join(output_dir, curr_vid + '.' + str(label)+'.txt')
    np.savetxt(output_file, start_end_table, fmt='%s')

if save_feats:
    for feats, label in feats_to_save:
        output_file     = os.path.join(output_dir, curr_vid + '.' + str(label)+'.npy')
        np.save(output_file, feats)

if save_person_faces:
    output_file     = os.path.join(output_dir, curr_vid + '_person_faces.npy')
    np.save(output_file, output_person_faces)