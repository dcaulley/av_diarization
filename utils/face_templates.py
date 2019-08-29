#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:43:07 2019

@author: dcaulley: Desmond Caulley

This script looks at all images in a directory and extracts VGGFace
feature vectors for faces in the images. It then uses DBSCAN to cluster
the faces and finds the largest cluster. This is the representative face
we keep. The output is average of all the features that constitutes largest cluster. 
"""
import os, glob
import subprocess
import argparse
import cv2, dlib
from imutils import face_utils

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

import tensorflow as tf
from keras import backend as K

import matplotlib.image as mpimg
from keras_vggface.vggface import VGGFace




parser = argparse.ArgumentParser(description = "creating template face using downloaded google images")

parser.add_argument('--img_dir',      type=str,   default='',                   help='directory to downloaded images'   )
parser.add_argument('--output_file',  type=str,   default='template_face.npy',  help='numpy file to store template face')
parser.add_argument('--avg_template', type=int,   default=1,                    help='set to 1 to return single vector')

opt = parser.parse_args();

img_dir      = opt.img_dir
output_file  = opt.output_file
avg_template = opt.avg_template #1 to return single vector which is average of cluster. 0 means return set of vectors that form largest cluster





def img_reformat(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    return img



def clustering_faces(features):
    clt = DBSCAN(metric="cosine", eps=0.4)
    clt.fit(features)
    
    labels   = clt.labels_
    try:
        indx = stats.mode(labels[np.where(labels != -1)])[0][0] #correct index using mode
        
        features = features[np.where(labels == indx)[0], :]
        
        return features
    except Exception as e:
        print(e)
        return



def template_feature(images):
    face_vectors = []
    for img in images:        
        try:
            img = mpimg.imread(img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            
            for (i, rect) in enumerate(faces):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                face = img[y:y+h, x:x+w]
                face = img_reformat(face)
                face_vectors.append(resnet50_features.predict(face))
        except Exception as e:
            print(e)
            continue
        
    
    face_vectors = np.concatenate(face_vectors, axis = 0)
    template_vectors = clustering_faces(face_vectors).T
    
    if template_vectors is not None:
        avg_template_vector = np.reshape(np.mean(template_vectors, axis=1), (-1, 1))
        
        if avg_template:
            template_vector = avg_template_vector
        else:
            template_vector = template_vectors
    
        return template_vector
    else:
        return

    
# Configuring to use only 80% of GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
K.set_session(session)


detector = dlib.get_frontal_face_detector()

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')
       
images = glob.glob(img_dir + '/*')
template_vector = template_feature(images)



# Saving Template Vector or error file creation if errors
if template_vector is not None:
    np.save(output_file, template_vector)
    print("\nDone Creating Template Vector for Celebrity\n")

else:
    err_dir  = os.path.join(os.path.split(output_file)[0], 'error_log')
    os.makedirs(err_dir, exist_ok=True)
    err_file = os.path.join(err_dir, 'face_template_error.txt')
    
    message = "Error creating template face. \nNeed more images and/or couldn't find cluster \
    with celebrity of interest"
    
    message  = '"%s"'%(message)
    err_file = '"%s"'%(err_file)
    
    command = 'echo %s > %s'%(message, err_file)
    subprocess.check_call(command, shell=True, stdout=None)



        



        