git clone https://github.com/tomole444/clean-pvnet
cd clean-pvnet
conda env create --file=environment.yml
conda activate pvnetclean
pip install Cython==0.28.2
pip install -r requirements.txt
pip install numpy
sudo apt-get install libglfw3-dev libglfw3
pip install -r requirements.txt
pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple/


#do compilation
ROOT=$(pwd)

echo "export PATH='/usr/local/cuda/bin:\$PATH'" >> ~/.bashrc
echo "export LD_LIBRARY_PATH='/usr/local/cuda/lib64:\$LD_LIBRARY_PATH'" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/lib/csrc/uncertainty_pnp/lib" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/lib/utils/extend_utils/lib" >> ~/.bashrc

cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda"
cd ransac_voting
python setup.py build_ext --inplace
cd ../nn
python setup.py build_ext --inplace
cd ../fps
python setup.py build_ext --inplace


# readjust version
pip install protobuf==3.18.3
pip install Pillow==6.1.0
pip install tensorboard
