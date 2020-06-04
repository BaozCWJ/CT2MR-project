import cv2
import skimage
import os
path = r'./raw/raw/'
i = 0
pathct = './jpg/ct/'
pathmri = './jpg/mri/'
for files in os.listdir(path):
    picpath = os.path.join(path,files)
    img = cv2.imread(picpath)
    ct = img[:,0:256,:]
    mri = img[:,256:512,:]
    cv2.imwrite(pathct + str(i) + '.jpg',ct)
    cv2.imwrite(pathmri + str(i) + '.jpg', mri)
    i += 1