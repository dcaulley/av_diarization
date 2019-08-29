# av_diarization

# possible downloads
First thing is to go on this website and download face landmark detection file and place in the utils/conf.
https://www.dropbox.com/sh/t5h024w0xkedq0j/AABS3GprqIvb_PwqeHOn2dxNa?dl=0&file_subpath=%2Fshape_predictor_194_face_landmarks.dat&preview=shape_predictor_194_face_landmarks.zip


# Running Code:
Call the script ./run_main.sh.
This script reads line by line a textfile of names and calls ./run_av_recipe.sh

You can run the code on a single person by directly calling ./run_av_recipe.sh args

# Possible error
error with finding TrackerCSRT() in face_verification_tracking.py. This has to do with opencv version


# Running Unsupervised
Need to call ./run_unsupervised_diarization.sh.

need to input a list of channels into that script for it to run

# Utilities
`check_total.py` - takes as input a directory with .txt files. Each .txt file contains IDed time segments for a particular celeb. This script sums up all the times in all the .txt files


`download_images.py` - takes as input a search term. Eg. "Elie Khoury face" and uses google_images_download api to download top results. User can specify how many images to download

`download_videos.py` - takes as input a search term. Eg. "Elie Khoury interview" and youtube-dl  to download top results. User can specify how many videos to download

`download_videos.py` - takes as input a search term. Eg. "Elie Khoury interview" and youtube-dl  to download top results. User can specify how many videos to download

`download_videos_channel.py` - takes as input a youtube channel url. It also takes in a match term, eg. "interview", so it can download only videos with that matchword in the title. User can specify the number of videos to download from the channel.

`face_templates.py` - takes as input directory to images. Uses VGG-face and DBSCAN clustering to find the most common face (mode face). Stores 2048-dimensional vector representing that face.

`face_to_times.py` - takes as input faces.pckl file. And returns total time in which the face of the person of interest has been identified. There is smoothing done to account for some of the discontinuities in detected face.

`face_verification_tracking.py` - takes as input a video, and also the templace face. It searches template-face in the video and uses CSRT based tracking to track the face. Face pixel histogram is used to assist tracking as well. Threshold value for 'cosine' similarity score set at 0.35.


`frame_rate_adjust.py` - changes frame rate fps of video to a set value.

`replay_face_verif_demo.py` - takes in two argument, the fps-adjusted video, and the obtained faces ID'd in the video. The output is a demo of the video playing with rectangular box surrounding the face.

`replay_unsupervised_faces_demo.py` - takes in two arguments, the fps-adjusted video, and the obtained faces from unsupervised-diarization-procedure stored as .npy file. The outcome is each ID'd person as unique 


