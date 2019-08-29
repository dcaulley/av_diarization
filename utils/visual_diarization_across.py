#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:23:18 2019

@author: dcaulley: Desmond Caulley

This script performs clustering across multiple videos. This script is heavily
dependent on having correct directory setup.
"""
import glob
import os, sys
import argparse
import subprocess

import numpy as np
from sklearn.cluster import AgglomerativeClustering


parser = argparse.ArgumentParser(description = "performs clustering and diarization across different videos")

parser.add_argument('--exp_dir',     type=str,   default='',                     help='directory to video_processing folder')
parser.add_argument('--output_dir',  type=str,   default='across_dir_output',    help='output directory')

opt = parser.parse_args();


os.makedirs(opt.output_dir, exist_ok=True)

data_files = glob.glob(opt.exp_dir + '/*/visual_diarization/*[0-9].npy')
text_files = glob.glob(opt.exp_dir + '/*/visual_diarization/*[0-9].txt')

data_files.sort()
text_files.sort()


text_labels = np.array([os.path.basename(k)[:-4] for k in text_files])
features = []
for feat_file in data_files:
    features.append(np.load(feat_file))

input_feats = np.array(features)


agg_cluster  = AgglomerativeClustering(n_clusters=None, affinity='cosine', \
                                       linkage='complete', distance_threshold=0.4)
predictions  = agg_cluster.fit_predict(input_feats).astype(int)

ided_files = []
for p in np.unique(predictions):
    pts = np.where(predictions == p)[0]
    np.random.shuffle(pts)
    file_ = os.path.basename(data_files[pts[0]])[:-4]
    idx   = np.where(text_labels == file_)[0][0]
    
    ided_files.append(text_files[idx])

# Saving IDed Text Files
for k in ided_files:
    command = "cp %s %s"%(k, opt.output_dir + '/.')
    subprocess.check_call(command, shell=True, stdout=None)