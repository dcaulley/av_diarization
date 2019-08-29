#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:57:03 2019

@author: dcaulley: Desmond Caulley

This script automatically downloads images form google. The inputs are simply the
search terms and the output directory to store results
"""
import os
import argparse
from google_images_download import google_images_download



parser = argparse.ArgumentParser(description = "downloads images using google api")

parser.add_argument('--search_term', type=str,   default='Elie Khoury face', help='eg. \"Elie Khoury face\"')
parser.add_argument('--num_images',  type=int,   default=25,                 help='num_images to download')
parser.add_argument('--output_dir',  type=str,   default='google_images',    help='dir to store images')

opt = parser.parse_args();


os.makedirs(opt.output_dir, exist_ok = True)

response  = google_images_download.googleimagesdownload()
arguments = {"keywords":opt.search_term, "limit":opt.num_images,  "output_directory": opt.output_dir, "no_directory":1}
paths     = response.download(arguments)

print("\nDone Downloading Images for Celebrity\n")