#!/usr/bin/python
from __future__ import print_function
import sys, time, os, pdb, argparse, pickle, subprocess


# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--data_dir', type=str, default='/dev/shm', help='Output direcotry');
parser.add_argument('--videofile', type=str, default='', help='Input video file');
parser.add_argument('--reference', type=str, default='', help='Name of the video');

#parser.add_argument('--fps', type=int, default=25, help='Frames per second for processing');

opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ========== ========== ========== ==========
# # EXECUTE DEMO
# ========== ========== ========== ==========

if not(os.path.exists(os.path.join(opt.work_dir,opt.reference))):
  os.makedirs(os.path.join(opt.work_dir,opt.reference))

if not(os.path.exists(os.path.join(opt.crop_dir,opt.reference))):
  os.makedirs(os.path.join(opt.crop_dir,opt.reference))

if not(os.path.exists(os.path.join(opt.avi_dir,opt.reference))):
  os.makedirs(os.path.join(opt.avi_dir,opt.reference))

if not(os.path.exists(os.path.join(opt.tmp_dir,opt.reference))):
  os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

vid_file = '"%s"'%opt.videofile
out_file = '"%s"'%os.path.join(opt.avi_dir,opt.reference,'video.avi')
FNULL=open(os.devnull,'w')
command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r %d -deinterlace %s" % (vid_file, 25, out_file)) #-async 1 
output = subprocess.call(command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

