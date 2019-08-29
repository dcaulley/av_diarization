#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:00:09 2019

@author: dcaulley: Desmond Caulley

Description: Takes in as argument text file with celebrity names
and stores the download images in provided output directory
"""

import os,sys
import argparse

from google_images_download import google_images_download


parser = argparse.ArgumentParser(description = "takes textfile with names and downloads google faces")

parser.add_argument('--celeb_list',  type=str,   default='',              help='input names text_file'   )
parser.add_argument('--output_dir',  type=str,   default='celeb_images',  help='output dir')
parser.add_argument('--num_images',  type=int,   default=25,              help='num of images to download per person')

opt = parser.parse_args();


def download_images(keyword, num_images, output_dir):
    response  = google_images_download.googleimagesdownload()
    
    arguments = {"keywords":keyword + ' face', "limit":num_images,  "output_directory": output_dir, "image_directory": keyword}
    response.download(arguments)
    


f_read     = open(opt.celeb_list, "r")
celeb_list = f_read.read().split('\n')
f_read.close()

os.makedirs(opt.output_dir, exist_ok=True)


for celeb in celeb_list[:-1]:
    download_images(celeb, opt.num_images, opt.output_dir)