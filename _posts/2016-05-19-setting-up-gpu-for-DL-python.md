---
layout: post
comments: true
title:  "Setting up laptop for Deep Learning using theano"
excerpt: "In this post I'll list out some steps you can follow to setup your machine for Deep Learning"
---

I recently purchased a new laptop with a NVIDIA GPU for building deep learning models. It has a NVIDIA Geforce GTX 960M GPU with around 600 cores and 4GB dedicated DDR5 memory. It faired good on theano benchmark test. I would suggest anyone looking for decent and affordable GPU's for deep learning to look for laptops with GTX GPUs.

But there were some issues which troubled me for days related to setting up GPU on ubuntu. In this post I'll write about those issues and some workaround them. Essentially this post is about setting up your laptop for deep learning in python.

Before getting started with the installation process I would like to comment on the version of ubuntu to choose. I tried to install nvidia driver on ubuntu 16.04 and I got lot of issues which were not addressed all on forums, this is most likely due to 16.05 being the recent release of ubuntu. In my opinion ubuntu 14.04 is the best because it's stable and you can find good support on online forums for it. 

1. Installing NVIDIA drivers :

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-352
```

Now this may or may not work. One common problem is presence of graphics driver "nouveau" in linux kernel itself. Sometimes it clashes with the driver being installed. 

People have posted many solutions for it like blacklisting this driver in configration file and so on. The issue is you cannot be sure which solution will work in your case. Unless you really know linux inside-out these are just brute force solutions which you can try. 

The workaround I found which was easy to incorporate and involved less "dirty" work was using bumblebee. This works if you have optimus laptop. Optimus laptops are latops which are shipped with both Intel and Nvidia GPUs. I think most laptops fall into this category. 

You can check if your laptop is one of them by running 

```
lspci | grep VGA
lspci | grep 3D
```

If you get an Intel card from 1st output and Nvidia card from 2nd output you are good to go.

Bumblee is a solution which helps to manage both GPU's on laptop. You can use Intel GPU for graphics stuff and NVIDIA card for scientific computing stuff.

To install NVIDIA drivers with bumblebee follow these steps :


```
sudo apt-get update && sudo apt-get upgrade
sudo apt-add-repository ppa:xorg-edgers/ppa 
sudo apt-get update
sudo apt-get upgrade && sudo apt-get dist-upgrade

sudo apt-get install bumblebee bumblebee-nvidia primus nvidia-352
sudo -H gedit /etc/modules 
```

Add these lines to it :

```
i915
bbswitch
```

```
sudo -H gedit /etc/bumblebee/bumblebee.conf
```

Add the following to line no. 22:
```
Driver=nvidia
```

on line 55:
```
KernelDriver=nvidia-352
```

on line 58:
```
LibraryPath=/usr/lib/nvidia-352:/usr/lib32/nvidia-352
```

on line 61:
```
XorgModulePath=/usr/lib/nvidia-349/xorg,/usr/lib/xorg/modules
```

Add following lines to file :
```
sudo -H gedit /etc/modprobe.d/bumblebee.conf
```

```
blacklist nvidia-352
```

Now reboot the machine. Test the installation using command :
```
nvidia-smi
```
If you are getting some error or this command is not showing up (command not found) then run the following commands :

```
sudo update-alternatives --config x86_64-linux-gnu_gl_conf
sudo ldconfig -n
sudo update-initramfs -u
```

After running first command you will be asked to select some file from multiple files. Select the one which looks similar to this :

/usr/lib/nvidia-352-prime/ld.so.conf

Again run nvidia-smi command to check if installtion is succesful or not. If you follow these steps nvidia-smi should show some GPU related stats.

2. Installing CUDA 

To install CUDA follow these steps

Download latest CUDA from [Nvidia](https://developer.nvidia.com/cuda-toolkit). Go to the folder where it is downloaded and run following commands :

```
sudo dpkg -i cuda-repo-ubuntu1404*amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

Add CUDA varibles to enviornment:
```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Restart your computer and check if CUDA is installed by running this command:

nvcc -V

3. Installing cuDNN :

To install cuDNN you only need to copy some files to cuda folder. Download [cuDNN](https://developer.nvidia.com/cudnn) and follow these commands :

```
tar xvf cudnn*.tgz
cd cuda
sudo cp */*.h /usr/local/cuda/include/
sudo cp */libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

4. Installing BLAS:

BLAS is matrix library with optimized matrix-vector and matrix-matrix operations. To install it run the following commands:

```
sudo apt-get install gfortran
mkdir ~/git
cd ~/git
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran -j $(($(nproc) + 1))
sudo make PREFIX=/usr/local install
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

5. Installing Anaconda:

Anaconda is package manager for installing python packages which are commonly used in data science applications. It includes jupyter-notebook, scipy, numpy, scklearn and so on.

To install anaconda, run the following commands:

```
wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh
bash Anaconda2-4.0.0-Linux-x86_64.sh
```

6. Installing theano:

```
sudo pip install Theano
```

Configure theano by creating ~/.theanorc config file and editing it according to information it's documentation.

To test installation fire-up python and type in following command 

```python
import theano
```

It should display cuDNN version in use and the make of GPU.

Now your laptop is ready to do some kick-ass deep learning :)

Cheers!!
