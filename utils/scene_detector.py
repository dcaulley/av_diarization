#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:12:34 2019

@author: dcaulley: Desmond Caulley

Detects scene boundaries in a video
"""

import os
import cv2

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors.content_detector import ContentDetector

def find_scenes(video_path):
    print(video_path)
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    video_framerate = video_manager.get_framerate()


    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).

    scene_manager.add_detector(ContentDetector(threshold=32))
    base_timecode = video_manager.get_base_timecode()


    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    scene_list = []
    scene_list_abbrev = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor()

        # Start video_manager.
        video_manager.start()

        print(video_manager.get(cv2.CAP_PROP_FRAME_COUNT))

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list(base_timecode)
        # Each scene is a tuple of (start, end) FrameTimecodes.


        print('List of scenes obtained:')
        print(scene_list)
        if len(scene_list) > 0:
          for i, scene in enumerate(scene_list):
              print(
                  'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                  i+1,
                  scene[0].get_timecode(), scene[0].get_frames(),
                  scene[1].get_timecode(), scene[1].get_frames(),))
              
              scene_list_abbrev.append(scene[1].get_frames())
        else:
          cap = cv2.VideoCapture(video_path)
          frame_num = 0

          while True:
            ret, image = cap.read()

            if ret == 0:
              break
            else:
              frame_num += 1

          scene_list_abbrev.extend([frame_num, frame_num])

          print('Entire video is one scene')

                    
            
        # Desmond Modification
        frames_read = scene_list_abbrev[-1]
        scene_list = scene_list_abbrev[:-1]

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return video_framerate, frames_read, scene_list