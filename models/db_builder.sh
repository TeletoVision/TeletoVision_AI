video=('cam_04' 'cam_05' 'cam_06' 'cam_07')

for i in "${!video[@]}"; do
    python db_builder.py --video_id ${video[i]}.mp4
done