import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score
import math

def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

root_path = './result/'
A_list = ['testA','fakeA_nice','fakeA_pix','fakeA_unit','fakeA_cyc']
B_list = ['testB','fakeB_nice','fakeB_pix','fakeB_unit','fakeB_cyc']
A_path = []
B_path = []
pic_out = np.array([])
for i in range(len(A_list)):
    A_path.append(pathlib.Path(root_path + A_list[i]))
    B_path.append(pathlib.Path(root_path + B_list[i]))

A_files = []
B_files = []
MSEA = [[],[],[],[]]
MSEB = [[],[],[],[]]
PSNRA = [[],[],[],[]]
PSNRB = [[],[],[],[]]
SSIMA = [[],[],[],[]]
SSIMB = [[],[],[],[]]
for i in range(5):
    A_files.append(list(A_path[i].glob('*.jpg')) + list(A_path[i].glob('*.png')))
    B_files.append(list(B_path[i].glob('*.jpg')) + list(B_path[i].glob('*.png')))

for i in range(5):
    A_files.append(os.listdir(root_path + A_list[i]))
    B_files.append(os.listdir(root_path + B_list[i]))
title = ['truth','NICE-GAN','pix2pix','UNIT','CycleGAN']
A_pics = np.array([])
B_pics = np.array([])
if not os.path.exists('./pltshow/'):
    os.mkdir('./pltshow/')
if not os.path.exists('./result/'):
    os.mkdir('./result/')
if not os.path.exists('./result/whole/'):
    os.mkdir('./result/whole/')
for i in range(len(A_files[0])):
    test_A = cv2.imread('./'+str(A_files[0][i]))
    test_B = cv2.imread('./'+str(B_files[0][i]))
    greytestA = cv2.cvtColor(test_A,cv2.COLOR_BGR2GRAY)
    greytestB = cv2.cvtColor(test_B, cv2.COLOR_BGR2GRAY)
    test_A = cv2.cvtColor(test_A,cv2.COLOR_BGR2RGB)
    test_B = cv2.cvtColor(test_B,cv2.COLOR_BGR2RGB)
    pic_out_A = test_A
    pic_out_B = test_B
    plt.figure()
    ax = plt.subplot(251)
    plt.axis('off')
    ax.set_title(title[0])
    plt.imshow(pic_out_A)
    ax = plt.subplot(256)
    plt.axis('off')
    ax.set_title(title[0])
    plt.imshow(pic_out_B)
    for j in range(1,len(A_list)):
        A = cv2.imread('./' + str(A_files[j][i]))
        greyA = cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
        A = cv2.cvtColor(A,cv2.COLOR_BGR2RGB)
        B = cv2.imread('./'+str(B_files[j][i]))
        greyB = cv2.cvtColor(B,cv2.COLOR_BGR2GRAY)
        B = cv2.cvtColor(B,cv2.COLOR_BGR2RGB)
        white = np.ones((pic_out.shape[0],20,3))*255
        #pic_out_A = np.concatenate((pic_out_A, white), axis=1)
        pic_out_A = np.concatenate((pic_out_A,A),axis=1)
        #pic_out_B = np.concatenate((pic_out_B, white), axis=1)
        pic_out_B = np.concatenate((pic_out_B,B),axis=1)
        ax = plt.subplot(251+j)
        ax.set_title(title[j])
        plt.axis('off')
        plt.imshow(A)
        if j<4:
            ax = plt.subplot(256+j)
        else:
            ax= plt.subplot(2,5,10)
        ax.set_title(title[j])
        plt.axis('off')
        plt.imshow(B)
        msea = mean_squared_error(greytestA,greyA)
        mseb = mean_squared_error(greytestB,greyB)
        MSEA[j-1].append(msea)
        MSEB[j-1].append(mseb)
        psnra = 10 * math.log10(255 ** 2 / msea)
        psnrb = 10 * math.log10(255 ** 2 / mseb)
        PSNRA[j-1].append(psnra)
        PSNRB[j-1].append(psnrb)
        SSIMA[j-1].append(ssim(greytestA,greyA))
        SSIMB[j-1].append(ssim(greytestB,greyB))
    cv2.imwrite('./result/whole/' + str(i+300) + '-A.jpg', pic_out_A)
    cv2.imwrite('./result/whole/' + str(i + 300) + '-B.jpg', pic_out_B)
    plt.savefig('./pltshow/'+str(i+300)+'.jpg')
    plt.close()
MSEA = np.array(MSEA)
MSEB = np.array(MSEB)
MSEA_std = np.std(MSEA,axis=1)
MSEB_std = np.std(MSEB,axis=1)
MSEA = np.mean(MSEA,axis=1)
MSEB = np.mean(MSEB,axis=1)
PSNRA = np.array(PSNRA)
PSNRB = np.array(PSNRB)
PSNRA_std = np.std(PSNRA,axis=1)
PSNRB_std = np.std(PSNRB,axis=1)
PSNRA = np.mean(PSNRA,axis=1)
PSNRB = np.mean(PSNRB,axis=1)
SSIMA = np.array(SSIMA)
SSIMB = np.array(SSIMB)
SSIMA_std = np.std(SSIMA,axis=1)
SSIMB_std = np.std(SSIMB,axis=1)
SSIMA = np.mean(SSIMA,axis=1)
SSIMB = np.mean(SSIMB,axis=1)
print('Mean Square Error-CT: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(MSEA[0],MSEA[1],MSEA[2],MSEA[3]))
print('Mean Square Error-MRI: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(MSEB[0],MSEB[1],MSEB[2],MSEB[3]))
print('PSNR-CT: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(PSNRA[0],PSNRA[1],PSNRA[2],PSNRA[3]))
print('PSNR-MRI: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(PSNRB[0],PSNRB[1],PSNRB[2],PSNRB[3]))
print('SSIM-CT: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(SSIMA[0],SSIMA[1],SSIMA[2],SSIMA[3]))
print('SSIM-MRI: NICE-GAN:{:.4f},PIX2PIX:{:.4f},UNIT:{:.4f},CycleGAN:{:.4f}'.format(SSIMB[0],SSIMB[1],SSIMB[2],SSIMB[3]))


print(MSEA_std)
print(MSEB_std)
print(PSNRA_std)
print(PSNRB_std)
print(SSIMA_std)
print(SSIMB_std)