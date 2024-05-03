import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def main():

    data_dir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchBig/"
    targetRes = tuple([1280,720])
    rgb_dir = os.path.join(data_dir, "rgb")
    pose_dir = os.path.join(data_dir, "pose")


    rgbImages = os.listdir(rgb_dir)
    poses = os.listdir(pose_dir)
    rgbImages.sort()
    poses.sort()

    for idx, rgb_img in enumerate(rgbImages):
        print(f"analysing {rgb_img} with pose {poses[idx]}")
        demo_image = cv2.imread(os.path.join(rgb_dir,rgb_img))
        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
        demo_image = cv2.resize(demo_image, targetRes)
        
        pose = np.load(os.path.join(pose_dir, poses[idx]))
        arr = np.array([0,0,0,1]).reshape((1,4))
        pose = np.vstack((pose, arr))
        print(pose)

        K = np.loadtxt(os.path.join(data_dir, "camera.txt"))

        resimg = demo_image.copy()
        resimg = draw_xyz_axis(resimg, pose, K = K, is_input_rgb= True)
        resimg = cv2.cvtColor(resimg, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(os.path.join(outdir, "pose" + rgb_img), resimg)
        cv2.imshow("Image", resimg)

        if cv2.waitKey(0) == ord("q"):
            break


def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
    '''
    @color: BGR
    '''
    if is_input_rgb:
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0,1]).astype(float)
    yy = np.array([0,1,0,1]).astype(float)
    zz = np.array([0,0,1,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
    xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
    yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
    zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
    line_type = cv2.FILLED
    arrow_len = 0
    tmp = color.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
    mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
    tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

    return tmp

def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  projected = K @ ((ob_in_cam@pt)[:3,:])
  projected = projected.reshape(-1)
  projected = projected/projected[2]
  return projected.reshape(-1)[:2].round().astype(int)

if __name__ == '__main__':
    #args.cfg_file =  "configs/custom.yaml" 
    #args.type =  "visualize"

    #ARGS: --type visualize --cfg_file configs/custom.yaml
    main()