#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/krdemo4', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,  opt.reference, 'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,  opt.reference, 'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir, opt.reference, 'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir, opt.reference, 'pycrop'))

# ==================== LOAD MODEL ====================

s = SyncNetInstance();

s.loadParameters('syncnet_python_rev/'+ opt.initial_model);
print("Model %s loaded."%opt.initial_model);

# ==================== GET OFFSETS ====================

with open(os.path.join(opt.work_dir,'tracks.pckl'), 'rb') as fil:
    tracks = pickle.load(fil)

dists = []
offsets = []
confs = []
frame_confs = []

for ii, track in enumerate(tracks):
    offset, conf, dist, frame_conf = s.evaluate(opt,videofile=os.path.join(opt.crop_dir,'%05d.avi'%ii))
    offsets.append(offset)
    dists.append(dist)
    confs.append(conf)
    frame_confs.append(frame_conf)
      
# ==================== PRINT RESULTS TO FILE ====================

with open(os.path.join(opt.work_dir,'offsets.txt'), 'w') as fil:
    fil.write('FILENAME\tOFFSET\tCONF\n')
    for ii, track in enumerate(tracks):
      fil.write('%05d.avi\t%d\t%.3f\n'%(ii, offsets[ii], confs[ii]))
      
with open(os.path.join(opt.work_dir,'activesd.pckl'), 'wb') as fil:
    pickle.dump(dists, fil)

with open(os.path.join(opt.work_dir,'frame_confs.pckl'), 'wb') as fil:
    pickle.dump(frame_confs, fil)
