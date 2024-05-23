#
export FPS=15
ffmpeg -f image2 -r $FPS -pattern_type glob -i '/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/result154/*.png' -b:v 40M -b:a 192k -vcodec mpeg4 -y /home/thws_robotik/Downloads/pvNet154.mp4 