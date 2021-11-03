# Server setup script
# Implemented by Jiahao Zhang

ssh mist@gpu166.mistgpu.xyz -p 31110

# prepare data
mkdir ~/data/
cp -v /data/Backgrounds_Validation.tar.gz ~/data/
cp -v /data/VideoMatte240K.tar.gz ~/data/
cd ~/data/
tar -zxvf ./Backgrounds_Validation.tar.gz ~/data/
tar -zxvf ./VideoMatte240K.tar.gz ~/data/

# prepare code
cd ~
git clone https://github.com/DavidZhang73/ENGN8501GroupProject.git

# install dependencies
sudo apt update
sudo apt install ffmpeg
sudo pip install kornia easing_functions

# preprocess data
## change the data path in extract_frames.py
mkdir ~/data/VideoMatte240KFrames/
cd ~/ENGN8501GroupProject/scripts
python ./extract_frames.py

# training
## change the data path in data_path.py
## change the hyper-parameters in
python train.py
tensorboard --logdir ./log --port 31114 --bind_all
http://gpu166.mistgpu.xyz:31114
