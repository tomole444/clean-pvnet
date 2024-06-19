from lib.config import cfg, args
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import socket
import pickle
from lib.utils.pvnet import pvnet_pose_utils
from lib.csrc.uncertainty_pnp import un_pnp_utils
import scipy
import scipy.linalg

confidences_sum = []
confidences_indiv = []

#infernce images from folder
def inference_folder():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    meta = np.load(args.meta, allow_pickle=True).item()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)

    #infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/ownBuchBlenderPVNet/rgb"
    infPath = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb"
    outdir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239_temp"
    viz_dir = os.path.join(outdir, "viz")
    pose_dir = os.path.join(outdir, "pose")
    mask_dir = os.path.join(outdir, "mask")
    os.makedirs(outdir, exist_ok= True)
    os.makedirs(viz_dir, exist_ok= True)
    os.makedirs(pose_dir, exist_ok= True)
    os.makedirs(mask_dir, exist_ok= True)

    upnp = True

    targetRes = tuple([1280,720])

    infImages = os.listdir(infPath)
    infImages.sort()
    #mean and stdw of image_net https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    #pyplotFig, pyplotAx = fig, ax = plt.subplots(1) 
    counter = 0

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
        if upnp:

            kpt_3d = np.array(meta["kpt_3d"])
            K = np.array(meta["K"])
            mask = output['mask'][0].detach().cpu().numpy()
            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
            var = output['var'][0].detach().cpu().numpy()

            pose_pred, _= uncertainty_pnp(kpt_3d, kpt_2d, var, K)
        else:
            pose_pred = visualizer.visualize_own(output, meta)#inp, meta, pyplotFig, pyplotAx)

            

        #plt.draw()
        #plt.pause(0.0001)
        #plt.savefig(os.path.join(outdir, infImg))
        resimg = demo_image.copy()
        resimg = draw_axis(resimg,pose_pred,K = meta["K"]) #draw_xyz_axis(resimg, pose_pred, K = meta["K"], is_input_rgb= True)
        resimg = cv2.cvtColor(resimg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(viz_dir,infImg), resimg)
        cv2.imwrite(os.path.join(mask_dir,infImg), mask * 255)

        T = np.vstack((pose_pred, np.array([0,0,0,1])))
        np.save(os.path.join(pose_dir, str(counter).zfill(5) + ".npy"),T)


        counter += 1
        # cv2.imshow("Image", resimg)

        # if cv2.waitKey(0) == ord("q"):
        #     break

        #pyplotAx.cla()
    
    confidences_sum_local = np.array(confidences_sum)
    confidences_indiv_local = np.array(confidences_indiv)
    save_arr = dict()
    save_arr["result_y"] = confidences_sum_local
    save_arr["ids"] = np.array(range(counter))
    save_arr = np.array(save_arr, dtype=object)
    np.save(os.path.join(outdir, "confidences_sum.npy"),save_arr, allow_pickle=True)
    save_arr = dict()
    save_arr["result_y"] = confidences_indiv_local
    save_arr["ids"] = np.array(range(counter))
    save_arr = np.array(save_arr, dtype=object)
    np.save(os.path.join(outdir, "confidences_indiv.npy"),save_arr, allow_pickle=True)

def inference_server():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    import matplotlib.patches as patches
    from lib.utils import img_utils


    meta = np.load(args.meta, allow_pickle=True).item()
    host= "192.168.99.91"#'localhost'
    port= 11024
    upnp = True

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
    img_out_dir = os.path.join("data", "inference")
    os.makedirs(img_out_dir, exist_ok= True)
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
        while True:
            print("Waiting for connection...")
            conn, addr = s.accept()
            print("connected with ", addr)
            with conn, conn.makefile('rb') as rfile:
                    img_count = 0
                    while True:
                        try:
                            print("waiting for packet...")
                            data = pickle.load(rfile)        
                        except EOFError: # when socket closes
                            break
                        #print(f'{addr}: {data}')
                        ret_info = dict()

                        image = data.copy()
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, targetRes)
                        inp = (((image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
                        #inp = (demo_image/255.).transpose(, 0, 1).astype(np.float32)
                        inp = torch.Tensor(inp[None]).cuda()
                        with torch.no_grad():
                            output = network(inp)

                        if upnp:
                            kpt_3d = np.array(meta["kpt_3d"])
                            K = np.array(meta["K"])
                            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
                            var = output['var'][0].detach().cpu().numpy()

                            pose_pred, confidence_indiv = uncertainty_pnp(kpt_3d, kpt_2d, var, K)
                            pose_pred = visualizer.visualize_own(output, meta) #dont use upnp pose
                        else:
                            pose_pred = visualizer.visualize_own(output, meta)#inp, meta, pyplotFig, pyplotAx)
                            confidence_indiv = [-1.0]
                        pose_pred = pose_pred.astype(np.float32)
                        
                        
                        
                        image = draw_xyz_axis(data.copy(), pose_pred, K = meta["K"], is_input_rgb= True)
                        cv2.imwrite(os.path.join(img_out_dir,str(img_count) + ".jpg"), image)
                        if(args.use_gui):
                            #inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
                            pyplotAx.imshow(image)
                            K = np.array(meta['K'])
                            corner_3d = np.array(meta['corner_3d'])
                            corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
                            #print(corner_3d)

                            #_, ax = plt.subplots(1)
                            pyplotAx.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
                            pyplotAx.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
                            plt.draw()
                            plt.pause(0.0001)

                        pose_pred = np.vstack((pose_pred, [0,0,0,1]))
                        #inverse position
                        #pose_pred = np.linalg.inv(pose_pred)
                        print(pose_pred)

                        ret_info["pose"] = pose_pred
                        confidence_indiv = np.sum(np.abs(confidence_indiv), axis=1)
                        confidence_indiv = confidence_indiv / 5
                        confidence_indiv = np.clip([confidence_indiv], 0.0, 1.0)
                        confidence_indiv = 1 - confidence_indiv 
                        ret_info["confidences"] = confidence_indiv

                        data = pickle.dumps(ret_info)
                        conn.send(data)

def uncertainty_pnp(kpt_3d, kpt_2d, var, K):
    cov_invs = []
    for vi in range(var.shape[0]):
        if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
            cov_invs.append(np.zeros([2, 2]).astype(np.float32))
        else:
            try:
                cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
                cov_invs.append(cov_inv)
            except:
                return np.identity(4), np.zeros((9,2,2))

    cov_invs = np.asarray(cov_invs)  # pn,2,2

    confidence_sum = np.sum(np.abs(cov_invs), axis=1)
    confidence_sum = np.sum(confidence_sum, axis=1)
    confidence_sum = np.sum(confidence_sum)
    confidences_sum.append(confidence_sum)

    weights = cov_invs.reshape([-1, 4])
    weights = weights[:, (0, 1, 3)]

    confidence_indiv = weights.copy()
    confidences_indiv.append(confidence_indiv)


    
    pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d, K)
    return pose_pred, confidence_indiv

def draw_axis(img, T, K):
    # unit is m
    #T = np.identity(4)
    #T[:3,3] = np.array([0.083,-0.3442,-1.097])
    rot_mat = T[:3,:3]
    rotV, _ = cv2.Rodrigues(rot_mat)
    tVec = T[:3,3]
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, tVec, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[0].ravel(), dtype=np.int16)), (0,0,255), 3)
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[1].ravel(), dtype=np.int16)), (0,255,0), 3)
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[2].ravel(), dtype=np.int16)), (255,0,0), 3)
    return img


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

def draw_pose_folder(pose_dir, rgb_dir, out_dir, K_path):
    K = np.loadtxt(K_path)
    pose_paths = os.listdir(pose_dir)
    rgb_paths = os.listdir(rgb_dir)

    pose_paths.sort()
    rgb_paths.sort()

    os.makedirs(out_dir, exist_ok=True)

    for idx, rgb_path in enumerate(rgb_paths):
        print("reading ", rgb_path)
        img = cv2.imread(os.path.join(rgb_dir,rgb_path))
        pose_compact = np.load(os.path.join(pose_dir,pose_paths[idx]))
        pose = np.identity(4)
        #pose[:3, :] = pose_compact
        pose = pose_compact

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose_img = draw_axis(img,pose,K)

        cv2.imwrite(os.path.join(out_dir, rgb_path), pose_img)

    




if __name__ == '__main__':
    #args.cfg_file =  "configs/custom.yaml" 
    #args.type =  "visualize"

    #ARGS: --type visualize --meta /home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/meta.npy --cfg_file configs/Leyh.yaml --use_gui 0 test.un_pnp True 
    
    #inference_folder()
    inference_server()
    #draw_pose_folder(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose", 
    #                 rgb_dir= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb", 
    #                 out_dir= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose_vis", 
    #                 K_path= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/cam_K.txt")