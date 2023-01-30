#!/bin/bash

# exit when any command fails
set -e

sudo apt-get update -y

# install conda
if ! conda -V; then
  wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
  sh ./Anaconda3-2019.07-Linux-x86_64.sh -b -p $HOME/anaconda3
  rm Anaconda3-2019.07-Linux-x86_64.sh
fi
eval "$($HOME/anaconda3/condabin/conda shell.bash hook)"
conda init
source ~/.bashrc

# install required ubuntu packages
sudo apt-get install -y \
  mysql-server \
  git git-lfs gcc \
  openssh-server \
  libnvidia-compute-430:amd64 \
  nvidia-utils-430 nvidia-settings nvidia-prime \
  nvidia-kernel-source-430 nvidia-kernel-common-430 \
  nvidia-driver-430 nvidia-dkms-430 nvidia-compute-utils-430 \
  ffmpeg xvfb

# needed for cuda
sudo apt-get install --reinstall build-essential

# allow $USER run sudo without password
if  ! sudo grep -qs "#### face-reidentification ####" /etc/sudoers; then
  echo "#### face-reidentification ####" | sudo tee -a /etc/sudoers
  echo "$USER      ALL=(ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers
  echo "#### face-reidentification ####" | sudo tee -a /etc/sudoers
fi

sudo mysql --execute="ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'detector'; FLUSH PRIVILEGES;"

eval "$($HOME/anaconda3/condabin/conda shell.bash hook)"
conda init
source ~/.bashrc

conda env create -n detector -f environment.yml
chmod +x start.sh
#sudo cp rc.local /etc/rc.local
#sudo chmod +x /etc/rc.loca

echo -e "\e[1mInstallation done: reboot now to proceed"
