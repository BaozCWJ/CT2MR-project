# About CT2MR project
## Introduction
some introduction
### Result
some comparisons
### Author
Sheng Hu & Weijie Chen

## Code structure
* main.py :  The standard workflow to build/train/test a model
* options.py : The hyperparameters of model and the setting about train/test process
* **util** : Some code about data visualization and  processing  for specific models
* **evaluation** : Some code about model evalution and data cleaning
* **data** : Some code about data loading and preprocess
* **models** : A folder where contains different models and network structure，include cycle-gan, pix2pix, nice-gan, unit and a nice-gan variant (mnice-gan).

### Contribution statement
Most of code in **util** and **data** credits to the projects in Acknowledgments.

In the folder **model**, we modified some code of 4 existed models to adapt the standard workflow. Besides, we made some improvement based on nice-gan and named it mnice-gan.


### Prerequisites
* Python 3.6.9
* Pytorch 1.1.0 and torchvision (https://pytorch.org/)
* TensorboardX
* Tensorflow (for tensorboard usage)
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.

## Usage
### Data prepare
You should arrange your dataset as the following folder structure
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg
           ├── bbb.png
           └── ...
       ├── testB
           ├── ccc.jpg
           ├── ddd.png
           └── ...
       └── label (only for nice-gan variant)
           ├── eee.jpg
           ├── fff.png
           └── ...
```

### Train
```
> python main.py --dataroot ./dataset/ct2mr --name gan1 --model nice_gan
```
* (For nice-gan and its variant) If the memory of gpu is **not sufficient**, set `--light`

### Test
```
> python main.py --dataroot ./dataset/ct2mr --phase test --name gan1 --model nice_gan
```
### Setting options
You can use the following command to know details of all setting options
```
> python main.py -h
```

## Acknowledgments
Our project contains the following model:
* cycle-gan and pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* nice-gan (https://github.com/alpc91/NICE-GAN-pytorch)
* UNIT (https://github.com/mingyuliutw/UNIT)
