# av_diarization

### Project Overview
The project can be split into two primary task. The first task is the case where the user provides a text file with a list of celebrites or persons of interest. For each person in the text, the system identifies times segments in a youtube video when that person is talking and stores the time segments as a text file. This task is a supervised search of a specific individual in a youtube video. The second task is unsupervised. This system was develop for the case where there might not be enough data for individuals in a particular language, eg. GA Language. In this scenario, the user provides a youtube channel where the people in that channel speak that relatively obscure language. The system then downloads videos from this channel and performs both intra-video clustering and inter-video clustering to obtain unique voices of various speakers.

- ### Supervised Audio Visual Diarization
  The input to this system is just a name. The system then follows the procedure outlined below and returns time segments in a youtube video when the individual is talking.
  - #### Step 1: Download Images
      The system takes in as input a name, eg. Desmond Caulley, and appends the word "face" or "photo" to it - i.e "Desmond Caulley photo." The system then performs automatic goole images search and downloads the top K results.
      ```
      Eg usage: utils/download_images.py --search_term="Desmond Caulley face" --num_images=25 --output_dir='~/output_folder'
      ```
   - #### Step 2: Face Templates
      The primary reasons for the images download is to use them to create a "template face" that will be used to compare to faces in a youtube video. The idea is that by searching for a person face using google-images. Most of the pictures returned will be the person of interest. There may be the pictures which belong to other people. To solve this issue, the pictures were download and Histogram of Oriented Gradients (HOG) based face detection was ran across all pictures. A VGG-face feature vector is extracted for each face. Lastly DBSCAN clustering was used to cluster faces with distance threshold set to 0.4. We assume the largest cluster after DBSAN will belong to the person of interest. We then average the vectors that belong to the largest cluster - this vector is what will be the template face.
      ```
      Eg usage: ./utils/face_templates.py --img_dir="download_images_dir --output_file="templat_face.npy'
      ```
    - #### Step 3: Download Youtube Videos
      The system takes in as input a name, eg. Desmond Caulley, and appends the word "interview" to it - i.e "Desmond Caulley interview." The system then uses youtube-dl to download top K video results from this search.
      ```
      Eg usage: ./utils/download_videos.py --search_term="Desmond Caulley interview" --num_videos=10 --output_dir='~/output_vid_folder'
      ```
      
    - #### Step 3: Face Detection, Verification, Tracking
      The next step is to find the template face in the youtube videos. For each youtube video, we first use pyscenedetect package to break down the video into various scenes and shots. For each scene, we run HOG face detection. We then compared each detected face to our template face for a match - verification. The threshold for verifying the template face with detected face is max_distance=0.35.  Once a face is verified, we track that face using OpenCV's implementation of CSRT. To make sure we are properly tracking a face, we take a histogram of the face when it's first detected. For each subsequent frame, we compare the histogram of the tracked face with the original histogram. Threshold for correct histogram tracking is set at 0.85.
      ```
      Eg usage: ./utils/face_verification_tracking.py --video_file="inputvid.mp4" --template_file="template_face.npy" --output_scenes="scenes.pckl" --output_faces="faces.pckl" --scoring_type="cosine" --threshold=0.35
      
    - #### Step 3: Smoothing and Crop Videos (Syncnet Implementation)
      Once the faces have been detected and tracked, we need to smooth out the boundary boxes of the face. This is a median smoothing of the temporal evolution of rectangle parameter for a face. Mini videos are cropped which will be passed into a syncnet.
      ```
      Eg usage: python ./syncnet_python_rev/run_pipeline.py --video_file="input_vid" --data_dir="output_dir" --reference="video_id" --scenes_file="scenes.pckl" --search_faces=0 --faces_file=faces.pckl" --min_track=25
      
    - #### Step 4: Syncnet
      Syncnet is used to determine offset between face and lips movement with the audio. If there is delay between audio and video, absolute value of the offset is 0. If the video doesn't match audio, offset is large. Additionally, sycnet outputs a confidence value. The higher the confidence value, the more confident sync_net is about the given offset.
      ```
      Eg usage: python ./syncnet_python_rev/run_syncnet_cuda.py --data_dir="output_dir" --reference="video_id"
      
   - #### Step 4: Post Processing
      The last step it processing the output of syncet and correctly identifying the time segments where syncet was confident there was audio-video synchronization. The output is a text file with times intervals where the particular person of interest was in a video and talking.
      ```
      Eg usage: ./utils/audio_visual_sync_match.py --exp_dir="exp_dir" --video_id="vid_name" --output_file="vid_name.txt






- ### Unsupervised Audio Diarization
  The input to this system is just a name. The system then follows the procedure outlined below and returns time segments in a youtube video when the individual is talking.
  - #### Step 1: Download Images
      The system takes in as input a name, eg. Desmond Caulley, and appends the word "face" or "photo" to it - i.e "Desmond Caulley photo." The system then performs automatic goole images search and downloads the top K results.
      ```
      Eg usage: utils/download_images.py --search_term="Desmond Caulley face" --num_images=25 --output_dir='~/output_folder'
      ```


### possible downloads
First thing is to go on this website and download face landmark detection file and place in the utils/conf.
https://www.dropbox.com/sh/t5h024w0xkedq0j/AABS3GprqIvb_PwqeHOn2dxNa?dl=0&file_subpath=%2Fshape_predictor_194_face_landmarks.dat&preview=shape_predictor_194_face_landmarks.zip


### Running Code:
Call the script ./run_main.sh.
This script reads line by line a textfile of names and calls ./run_av_recipe.sh

You can run the code on a single person by directly calling ./run_av_recipe.sh args

### Possible error
error with finding TrackerCSRT() in face_verification_tracking.py. This has to do with opencv version


### Running Unsupervised
Need to call ./run_unsupervised_diarization.sh.

need to input a list of channels into that script for it to run

### Utilities
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


