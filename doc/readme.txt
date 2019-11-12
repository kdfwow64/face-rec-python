# pc1 ssh port forward via proxy
/usr/bin/ssh -i /home/user/.ssh/id_rsa -NT -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -R 9091:localhost:22 root@165.22.75.63

# pc2 ssh port forward via proxy
/usr/bin/ssh -i /home/user/.ssh/id_rsa -NT -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -R 9092:localhost:22 root@165.22.75.63


# vnc port forward
/usr/bin/ssh -i /home/user/.ssh/id_rsa -NT -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes -R 5901:localhost:5901 root@165.22.75.63


# ssh to pc1
ssh -p 9091 root@165.22.75.63 -i ~/.ssh/id_rsa -l user


# ssh to pc2
ssh -p 9092 root@165.22.75.63 -i ~/.ssh/id_rsa -l user



# record from cam1 and cam2 in 30 fps
# https://stackoverflow.com/questions/44960632/ffmpeg-records-5-frames-per-second-on-a-device-that-cheese-records-at-20-fps
ffmpeg -y -f video4linux2 -video_size 1280x720 -r 30 -c:v mjpeg -i /dev/video0 -c:v copy cam1.mp4
ffmpeg -y -f video4linux2 -video_size 1280x720 -r 30 -c:v mjpeg -i /dev/video2 -c:v copy cam2.mp4

# record from cam1 and cam2 in 30 fps, but encode with h264 (way less space!!)
ffmpeg -y -f video4linux2 -video_size 1280x720 -r 30 -c:v mjpeg -i /dev/video0  cam1.mp4
ffmpeg -y -f video4linux2 -video_size 1280x720 -r 30 -c:v mjpeg -i /dev/video2  cam2.mp4

# cut the video based on time
ffmpeg -i in_movie.mp4 -ss 00:00:03 -t 00:00:08 -async 1 cut.mp4

# Classifier training of inception resnet v1
https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1