
## Usage
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
       └── testB
           ├── ccc.jpg
           ├── ddd.png
           └── ...
```
### Prerequisites
* Python 3.6.9
* Pytorch 1.1.0 and torchvision (https://pytorch.org/)
* TensorboardX
* Tensorflow (for tensorboard usage)
* CUDA 10.0.130, CuDNN 7.3, and Ubuntu 16.04.


### Train
```
> python main.py --dataroot ./dataset/cat2dog --name gan1 --model nice_gan
```
* (For nice-gan) If the memory of gpu is **not sufficient**, set `--light` to True

### Test
```
> python main.py --dataroot ./dataset/cat2dog --phase test --name gan1 --model nice_gan
```


## Acknowledgments
Our project contains the following model:
* cycle-gan and pix2pix (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* nice-gan (https://github.com/alpc91/NICE-GAN-pytorch)
