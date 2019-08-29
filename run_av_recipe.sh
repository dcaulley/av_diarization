#!/bin/bash
export OMP_NUM_THREADS=10

gpu=$1
celeb="$2"
exp_dir="$3"
face_word=$4
interview_word=$5

exp_dir="$exp_dir/$celeb"

vid_pro=$exp_dir/video_processing

num_images=25
num_videos=5
fps=25


stage=1

echo "$celeb"


if [ -d "$vid_pro" ] && [ -f "$exp_dir"/curr_working.lst ]; then
	exit 0
fi


if [ -f "$exp_dir"/done.lst ]; then
	exit 0
fi

mkdir -p "$vid_pro"
touch "$exp_dir"/curr_working.lst




if [ $stage -le 1 ]; then
	# Download images celebrity; create template face using DBClustering
	if [ ! -f "$exp_dir"/template_face.npy ]; then
		CUDA_VISIBLE_DEVICES=$gpu ./utils/download_images.py --search_term "$celeb $face_word" --num_images $num_images --output_dir "$exp_dir/images"
		CUDA_VISIBLE_DEVICES=$gpu ./utils/face_templates.py  --img_dir "$exp_dir/images"  --output_file "$exp_dir"/template_face.npy
	fi
fi
echo; echo "Stage 1 Complete: Images Downloaded and Template Vector Created"; echo





if [ $stage -le 2 ] && [ ! -d "$exp_dir/videos" ] ; then

	# Downloading videos from youtube for the celebrity. Saved using video ID; fps adjusted to 25
	CUDA_VISIBLE_DEVICES=$gpu ./utils/download_videos.py  --search_term "$celeb $interview_word"  --num_videos $num_videos --output_dir "$exp_dir/videos" --fps $fps
fi
echo; echo "Stage 2 Complete: Downloaded all Youtube Videos"; echo





if [ $stage -le 8 ]; then
	if [ ! -f "$exp_dir"/template_face.npy ]; then
		touch "$exp_dir"/done.lst
		exit 0
	fi
	

	for vid in "$exp_dir/videos"/*.mp4
	do

		vid_name="$(basename "$vid" .mp4)"
		vid_dir="$vid_pro"/"$vid_name"

		echo "$vid_name"


		typeset -i total_time=$( ./utils/check_total.py "$exp_dir" )

		if [ "$total_time" -gt "500" ] || [ -f "$exp_dir"/output_vid.avi ]; then
			break
		fi


		if  [ $stage -le 3 ] && [ ! -f "$vid_pro"/"$vid_name"/pywork/tracks.pckl ]; then
			CUDA_VISIBLE_DEVICES=$gpu python ./syncnet_python_rev/run_init.py --videofile "$vid" --data_dir "$vid_pro" --reference="$vid_name"
		fi



		if [ $stage -le 4 ] && [ ! -f "$vid_dir"/pywork/faces.pckl ]; then
			CUDA_VISIBLE_DEVICES=$gpu ./utils/face_verification_tracking.py  --video_file "$vid_dir"/pyavi/video.avi \
									 --template_file "$exp_dir"/template_face.npy	--output_scenes "$vid_dir"/pywork/scenes.pckl \
									 --output_faces "$vid_dir"/pywork/faces.pckl
		fi



		if [ $stage -le 5 ] && [ ! -f "$vid_pro"/"$vid_name"/pywork/tracks.pckl ]; then

			CUDA_VISIBLE_DEVICES=$gpu python ./syncnet_python_rev/run_pipeline.py --video_file "$vid" --data_dir "$vid_pro" --reference="$vid_name" \
											--scenes_file "$vid_pro"/"$vid_name"/pywork/scenes.pckl --search_faces 0 \
											--faces_file "$vid_pro"/"$vid_name"/pywork/faces.pckl --min_track 25


			echo; echo "Completed Face Tracking Procedure"; echo
		fi



		if [ $stage -le 6 ] && [ ! -f "$vid_pro"/"$vid_name"/pywork/offsets.txt ]; then

			CUDA_VISIBLE_DEVICES=$gpu python ./syncnet_python_rev/run_syncnet_cuda.py --data_dir "$vid_pro" --reference="$vid_name" 

			echo; echo "Completed Face Tracking Procedure"; echo
		fi




		if [ $stage -le 7 ]; then

			CUDA_VISIBLE_DEVICES=$gpu ./utils/audio_visual_sync_match.py --exp_dir "$vid_pro" --video_id="$vid_name" --output_file "$exp_dir"/"$vid_name".txt
			echo; echo "Completed Audio Visual Match"; echo
		fi

		echo; echo "Completed Video" "$vid_name"; echo

	done
fi


if [ $stage -le 10 ] && [ ! -f "$exp_dir"/output_vid.avi ]; then

	for vid in "$exp_dir"/*.txt
	do
		vid_name="$(basename "$vid" .txt)"
		echo $vid_name
		if  [ ! -f "$vid_pro"/"$vid_name"/pyavi/video.avi ]; then
			CUDA_VISIBLE_DEVICES=$gpu python ./syncnet_python_rev/run_init.py --videofile="$exp_dir/videos/$vid_name.mp4" \
				--data_dir "$vid_pro" --reference="$vid_name"
		fi
	done


	./utils/video_stitching_all.py "$exp_dir" "$exp_dir"/output_vid.avi
fi


rm    "$exp_dir"/curr_working.lst; touch "$exp_dir"/done.lst

rm "$vid_pro"/*/pyavi/video.avi
exit 0
