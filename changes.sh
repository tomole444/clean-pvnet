#setup
git clone https://github.com/tomole444/clean-pvnet.git
cd clean-pvnet

conda env create -f environment.yml

conda activate pvnetclean

pip install Cython==0.28.2
sudo apt-get install libglfw3-dev libglfw3

pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple/

# for compute arch higher 8.8 torch===1.10.0+cu113 torchvision===0.11.0+cu113 torchaudio===0.10.0+cu113

pip install -r requirements_docker.txt

ROOT=$(pwd)
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda"
cd ransac_voting
python setup.py build_ext --inplace
cd ../nn
python setup.py build_ext --inplace
cd ../fps
python setup.py build_ext --inplace

# conda env
conda activate pvnetclean 

# updated the torch version to 1.7.1 to prevent the THC.cu issue
pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple/
#downgrade protobuf
pip install protobuf==3.18.3
#Data structure
#├── /path/to/dataset
#│   ├── model.ply
#│   ├── camera.txt
#│   ├── diameter.txt  // the object diameter, whose unit is meter
#│   ├── rgb/
#│   │   ├── 0.jpg
#│   │   ├── ...
#│   │   ├── 1234.jpg
#│   │   ├── ...
#│   ├── mask/
#│   │   ├── 0.png
#│   │   ├── ...
#│   │   ├── 1234.png
#│   │   ├── ...
#│   ├── pose/
#│   │   ├── pose0.npy
#│   │   ├── ...
#│   │   ├── pose1234.npy
#│   │   ├── ...
#│   │   └──

#Pose is 3x4 with rotmat and transvec

#compute FPS keypoints
python run.py --cfg_file configs/Leyh.yaml --type custom_split

# training on custom dataset
python train_net.py --cfg_file configs/Leyh.yaml train.batch_size 2

# start evaluation
python train_net.py --cfg_file configs/Leyh.yaml --test True 

#monitor progress
tensorboard --logdir data/record/pvnet --bind_all

#visualize infernce
python run.py --type visualize --cfg_file configs/Leyh.yaml


#inference
python inference.py --type visualize --cfg_file configs/Leyh.yaml
