import numpy as np
import os

def main():
    kpt_3d = np.loadtxt("data/custom/kpt_3d.txt")
    corner_3d = np.loadtxt("data/custom/corner_3d.txt")
    K = np.loadtxt("data/custom/camera.txt")
    arr = dict()
    arr["kpt_3d"] = kpt_3d
    arr["corner_3d"] = corner_3d
    arr["K"] = K
    arr = np.array(arr, dtype=object)
    np.save("/home/thws_robotik/Documents/Leyh/6dpose/detection/clean-pvnet/data/custom/meta.npy",arr, allow_pickle=True)

if __name__ == "__main__":
    main()
