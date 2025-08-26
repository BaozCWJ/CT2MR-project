from skimage import io,transform
import glob
import os
import numpy as np

#数据集地址


w=256
h=256


def read_image(path,img_type):
    data_path = path+img_type
    cate=[data_path+x for x in os.listdir(data_path) if os.path.isdir(data_path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            #print('reading the images:%s'%(im))
            img=io.imread(im,as_gray=True)
            img=transform.resize(img,(w,h))
            #print(img.shape)
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32).reshape((-1,w,h,1)),np.asarray(labels,np.int32).reshape((-1,1))




#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt].reshape((-1,1))


def index4_4(labels,predictions):
    TP = np.sum(predictions*labels.T)
    TN = np.sum((1-predictions)*(1-labels).T)
    FP = np.sum(predictions*(1-labels).T)
    FN = np.sum((1-predictions)*labels.T)
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = (TP)/(TP+FP)
    rec = (TP)/(TP+FN)
    F1 = 2*prec*rec/(prec+rec)
    return [TP,TN,FP,FN,acc,prec,rec,F1]
