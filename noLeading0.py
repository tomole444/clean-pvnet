import numpy as np
import os


path = '/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchPvnet/pose'
poses = os.listdir(path)
poses.sort()

for entry in poses:
  print(f"reading {entry}")
  #if not entry.endswith(".jpg") and not entry.endswith(".png"):
  #  continue
  entrySplit = entry.split(".")
  name = str(int(entrySplit[0]))
  ext = entrySplit[1]
  filename = name + "." + ext
  print(f"saving  {filename}")
  os.rename(os.path.join(path, entry), os.path.join(path, filename))