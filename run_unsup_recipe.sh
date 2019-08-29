#!/bin/bash
export OMP_NUM_THREADS=10

ch1_url='https://www.youtube.com/user/nessmatvreplays'
ch2_url='https://www.youtube.com/user/NessmaTVReplay'
ch3_url='https://www.youtube.com/user/EttounsiaReplay'
ch4_url='https://www.youtube.com/user/televisiontunisie6'


gpu=$1
exp_dir=$2


match_word=interview

vid_pro=$exp_dir/video_processing

num_videos=5
min_time=300
fps=25



stage=1

if [ $stage -le -1 ] ; then
	# Downloading videos from youtube for the celebrity. Saved using video ID
	# Modifying video to have frame rate of 25 fps.
	./utils/download_videos_channel.py  --youtube_channel="$ch1_url"  --match_term=$match_word --num_videos=$num_videos --output_dir="$exp_dir/videos" --fps=$fps
	./utils/download_videos_channel.py  --youtube_channel="$ch2_url"  --match_term=$match_word --num_videos=$num_videos --output_dir="$exp_dir/videos" --fps=$fps
	./utils/download_videos_channel.py  --youtube_channel="$ch3_url"  --match_term=$match_word --num_videos=$num_videos --output_dir="$exp_dir/videos" --fps=$fps
	./utils/download_videos_channel.py  --youtube_channel="$ch4_url"  --match_term=$match_word --num_videos=$num_videos --output_dir="$exp_dir/videos" --fps=$fps
	./utils/download_videos_channel.py  --youtube_channel="$ch1_url"  --match_term=$match_word --num_videos=$num_videos --output_dir="$exp_dir/videos" --fps=$fps

	echo; echo "Stage 2 Complete: Downloaded all Youtube Videos"; echo
fi



if [ $stage -le 2 ]; then

	for vid in "$exp_dir/videos"/*.mp4
	do

		vid_name="$(basename "$vid" .mp4)"
		echo $vid_name

		if [ -d "$vid_pro"/"$vid_name"/visual_diarization ]; then
			continue
		fi



		if [ $stage -le 3 ] && [ ! -d "$vid_pro"/"$vid_name"/pywork ]; then

			CUDA_VISIBLE_DEVICES=$gpu python ./syncnet_python_rev/run_pipeline.py --video_file="$vid" --data_dir="$vid_pro" --reference="$vid_name" \
																--search_faces=1


			echo; echo "Video Processing"; echo
		fi

		if [ $stage -le 4 ]; then

			CUDA_VISIBLE_DEVICES=$gpu ./utils/visual_diarization_single.py --exp_dir="$vid_pro" --curr_vid="$vid_name" --output_dir="$vid_pro"/"$vid_name"/visual_diarization --min_time=200


			echo; echo "$vid_name"; echo
		fi
	done

fi



if [ $stage -le 5 ]; then

  CUDA_VISIBLE_DEVICES=$gpu ./utils/visual_diarization_across.py --exp_dir="$vid_pro" --output_dir="$exp_dir"/unsup_output


  echo; echo "$vid_name"; echo
fi
