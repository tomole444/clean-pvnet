import numpy as np
import os


#pose = np.load("/home/thws_robotik/Downloads/pose1178.npy")
pose = np.loadtxt("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuch/ob_in_cam/02274.txt")
print(pose[:3,:])