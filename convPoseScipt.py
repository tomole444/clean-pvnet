import numpy as np
import os


path = '/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuch/ob_in_cam'
outPath = '/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchPvnet/pose'
poses = os.listdir(path)
poses.sort()

os.makedirs(outPath, exist_ok= True)
for entry in poses:
  print(f"reading {entry}")
  if not entry.endswith(".txt"):
    continue
  frame = np.loadtxt(os.path.join(path,entry))
  frame = frame[:3,:]
  #frame = cv2.resize(frame, (2560, 1440)) 
  destfile = entry.replace(".txt",".npy")
  print(f"saving  {destfile}")
  np.save(os.path.join(outPath, destfile), frame)