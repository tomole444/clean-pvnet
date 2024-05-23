from lib.config import cfg, args
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import socket
import pickle
from lib.utils.pvnet import pvnet_pose_utils

#infernce images from folder
def inference_folder():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    meta = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/clean-pvnet/data/custom/meta.npy", allow_pickle=True).item()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)

    #infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchBlenderPVNet/rgb"
    infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/color"
    outdir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/color/resultBig239"
    os.makedirs(outdir, exist_ok= True)
    targetRes = tuple([1280,720])

    infImages = os.listdir(infPath)
    infImages.sort()
    #mean and stdw of image_net https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    pyplotFig, pyplotAx = fig, ax = plt.subplots(1) 


    for infImg in infImages:
        print(f"analysing {infImg}")
        demo_image = cv2.imread(os.path.join(infPath,infImg))
        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
        demo_image = cv2.resize(demo_image, targetRes)
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        #inp = (demo_image/255.).transpose(, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        pose_pred = visualizer.visualize_own(output, inp, meta, pyplotFig, pyplotAx)

        #plt.draw()
        #plt.pause(0.0001)
        #plt.savefig(os.path.join(outdir, infImg))
        resimg = demo_image.copy()
        resimg = draw_xyz_axis(resimg, pose_pred, K = meta["K"], is_input_rgb= True)
        resimg = cv2.cvtColor(resimg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(outdir, "pose" + infImg), resimg)

        # cv2.imshow("Image", resimg)

        # if cv2.waitKey(0) == ord("q"):
        #     break

        pyplotAx.cla()

def inference_server():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer


    meta = np.load(args.meta, allow_pickle=True).item()
    host= 'localhost'
    port= 11024
    pvnet_termination_string = b'\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01\x00\x01'

    network = make_network(cfg).cuda()
    
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)

    #infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchBlenderPVNet/rgb"
    # infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/color"
    # outdir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBookInference/color/resultBig239"

    #os.makedirs(outdir, exist_ok= True)
    targetRes = tuple([1280,720])

    # infImages = os.listdir(infPath)
    # infImages.sort()
    #mean and stdw of image_net https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    if args.use_gui:
        pyplotFig, pyplotAx = fig, ax = plt.subplots(1)
    else:
        pyplotFig, pyplotAx = None, None


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f'Server listening on {host}:{port}')
        
        conn, addr = s.accept()
        with conn, conn.makefile('rb') as rfile:
                while True:
                    try:
                        print("waiting for packet...")
                        data = pickle.load(rfile)        
                    except EOFError: # when socket closes
                        break
                    #print(f'{addr}: {data}')
                    image = data
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, targetRes)
                    inp = (((image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
                    #inp = (demo_image/255.).transpose(, 0, 1).astype(np.float32)
                    inp = torch.Tensor(inp[None]).cuda()
                    with torch.no_grad():
                        output = network(inp)
                    pose_pred = visualizer.visualize_own(output, meta)
                    pose_pred = pose_pred.astype(np.float32)
                    pose_pred = np.vstack((pose_pred, [0,0,0,1]))
                    if(args.use_gui):
                        # inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
                        # pyplotAx.imshow(inp)
                        # corner_3d = np.array(meta['corner_3d'])
                        # corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
                        # #print(corner_3d)

                        # #_, ax = plt.subplots(1)
                        # pyplotAx.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
                        # pyplotAx.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
                        plt.draw()
                        plt.pause(0.0001)
                    data = pickle.dumps(pose_pred)
                    conn.send(data)




def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
    '''
    @color: BGR
    '''
    if is_input_rgb:
        color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
    xx = np.array([1,0,0]).astype(float)
    yy = np.array([0,1,0]).astype(float)
    zz = np.array([0,0,1]).astype(float)
    xx[:3] = xx[:3]*scale
    yy[:3] = yy[:3]*scale
    zz[:3] = zz[:3]*scale
    
    origin = tuple(pvnet_pose_utils.project(np.array([0,0,0]),K,ob_in_cam).reshape(-1).astype(np.int16))
    xx = tuple(pvnet_pose_utils.project(xx,K,ob_in_cam).reshape(-1).astype(np.int16))
    yy = tuple(pvnet_pose_utils.project(yy,K,ob_in_cam).reshape(-1).astype(np.int16))
    zz = tuple(pvnet_pose_utils.project(zz,K,ob_in_cam).reshape(-1).astype(np.int16))
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

#deprecated
def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  projected = K @ ((ob_in_cam@pt)[:3,:])
  projected = projected.reshape(-1)
  projected = projected/projected[2]
  return projected.reshape(-1)[:2].round().astype(int)

if __name__ == '__main__':
    #args.cfg_file =  "configs/custom.yaml" 
    #args.type =  "visualize"

    #ARGS: --type visualize --cfg_file configs/Leyh.yaml
    #inference_folder()
    inference_server()