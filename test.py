import numpy as np
import os


#pose = np.load("/home/thws_robotik/Downloads/pose1178.npy")
#pose = np.loadtxt("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuch/ob_in_cam/02274.txt")
pose = np.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchPvnet/pose/pose0.npy")
kpt = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/clean-pvnet/demo_images/cat/meta.npy", allow_pickle=True).item()
kpt2 = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/clean-pvnet/data/custom/meta.npy", allow_pickle=True).item()
#kpt = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/clean-pvnet/demo_images/cat/meta.npy", allow_pickle=True)
#print(pose[:3,:])
print("test")