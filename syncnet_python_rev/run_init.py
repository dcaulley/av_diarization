#!/usr/bin/python
from __future__ import print_function
import sys, time, os, pdb, argparse, pickle, subprocess

import cv2
import tensorflow as tf
from keras import backend as K
# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir', type=str, default='/dev/shm', help='Output direcotry');
parser.add_argument('--videofile', type=str, default='', help='Input video file');
parser.add_argument('--reference', type=str, default='', help='Name of the video');

#parser.add_argument('--fps', type=int, default=25, help='Frames per second for processing');

opt = parser.parse_args();


setattr(opt,'avi_dir',os.path.join(opt.data_dir,  opt.reference, 'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,  opt.reference, 'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir, opt.reference, 'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir, opt.reference, 'pycrop'))


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========


if not(os.path.exists(os.path.join(opt.work_dir))):
  os.makedirs(os.path.join(opt.work_dir))

if not(os.path.exists(os.path.join(opt.crop_dir))):
  os.makedirs(os.path.join(opt.crop_dir))

if not(os.path.exists(os.path.join(opt.avi_dir))):
  os.makedirs(os.path.join(opt.avi_dir))

if not(os.path.exists(os.path.join(opt.tmp_dir))):
  os.makedirs(os.path.join(opt.tmp_dir))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.23
session = tf.Session(config=config)
K.set_session(session)

vid_file = '"%s"'%opt.videofile
out_file = '"%s"'%os.path.join(opt.avi_dir,'video.avi')
FNULL=open(os.devnull,'w')

cap = cv2.VideoCapture(opt.videofile)
fps = round(cap.get(5))
if cap.get(5) >= 25:
	fps = 25

command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r %d -deinterlace %s" % (vid_file, fps, out_file)) #-async 1 
output  = subprocess.call(command, shell=True, stderr=subprocess.STDOUT)

command = "echo %d > %s"%(fps, '"%s"'%os.path.join(opt.work_dir, 'fps.txt'))
output  = subprocess.call(command, shell=True, stderr=subprocess.STDOUT)

