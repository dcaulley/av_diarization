list_celeb=$1 #file with celeb names listed. eg. celeb_names.lst
gpu=$2

exp_dir=$3 #eg exp_tunisian

face_word=photo #search terms for face
interview_word=interview #search term for interview


while read line; do 
	name="$line"
	./run_av_recipe.sh $gpu "$name" $exp_dir $face_word $interview_word </dev/null
done<$list_celeb
