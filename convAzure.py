import cv2
import os

targetRes = (1280,720)


path = '/home/thws_robotik/Documents/Leyh/azure_kinect/out/depth'
outPath = '/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/depth'
imgs = os.listdir(path)
imgs.sort()

os.makedirs(outPath, exist_ok= True)
for entry in imgs:
    print(f"reading {entry}")
    frame = cv2.imread(path + "/" + entry, cv2.IMREAD_UNCHANGED)
    #frame = cv2.resize(frame, (2560, 1440)) 
    destfile = entry.replace(".jpg",".png")
    print(f"saving  {destfile}")
    frame = cv2.resize(frame, targetRes)
    cv2.imwrite(outPath + "/" + destfile, frame)