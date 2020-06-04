from torchvision import transforms,models
import numpy as np
from torch.autograd import Variable
import torch
from sklearn.metrics.pairwise import polynomial_kernel
from dataset2 import dataset2
from torch.utils.data import DataLoader
import torch.nn as nn
class IV3(nn.Module):
    def __init__(self):
        super(IV3,self).__init__()
        inception = models.inception_v3(pretrained=True)
        inception.fc = nn.Sequential()
        self.net = inception
    def forward(self,x):
        y = self.net(x)
        return y


def tezhen(pred,cuda=False):
    model = IV3()
    model.eval()
    if pred.shape[2]==3:
        tran = transforms.ToTensor()
        pred = tran(pred)
        pred = Variable(torch.unsqueeze(pred, dim=0).float(), requires_grad=False)
        #truth = tran(truth)
        #truth = Variable(torch.unsqueeze(truth, dim=0).float(), requires_grad=False)
    if cuda:
        model = model.cuda()
        pred = pred.cuda()
        #truth = truth.cuda()
    pred_act = model(pred)
    #truth_act = model(truth)
    return pred_act
def cal_fid(act1,act2,eps=1e-10):
    miu1 = np.mean(act1,axis=0)
    miu2 = np.mean(act2,axis=0)
    cov1 = np.cov(act1,rowvar=False)
    cov2 = np.cov(act2,rowvar=False)
    mu = miu1 - miu2
    u,sigma,v = np.linalg.svd(cov1.dot(cov2))
    sigma[sigma<eps] = 0
    sigma = np.sqrt(sigma)
    cov_sqrt = np.dot(u*sigma,v)
    return mu.dot(mu)+ np.trace(cov1 + cov2 - 2*cov_sqrt)

def cal_kid(act1,act2):
    m = act1.shape[0]
    n = act2.shape[0]
    k11 = polynomial_kernel(act1,degree = 3, gamma = 1/2048, coef0 = 1)
    k22 = polynomial_kernel(act2,degree = 3, gamma = 1/2048, coef0 = 1)
    k12 = polynomial_kernel(act1,act2,degree = 3, gamma = 1/2048, coef0 = 1)
    k11sum = np.sum(k11-np.diag(np.diag(k11)))/(m*(m-1))
    k22sum = np.sum(k22 - np.diag(np.diag(k22)))/(n*(n-1))
    k12sum = np.sum(k12)/(m*n)
    return k11sum+k22sum-2*k12sum


if __name__ == '__main__':
    act1 = np.random.rand(64,2048)
    act2 = np.random.rand(64,2048)
    print (cal_kid(act1,act2))
    batch_size = 5
    fake = dataset2(image_path='./result/fakeA_nice/')
    fake_loader = DataLoader(dataset=fake,batch_size=batch_size)
    truth = dataset2(image_path='./result/testA/')
    truth_loader = DataLoader(dataset=truth,batch_size=batch_size)
    act1 = np.zeros((367,2048))
    act2 = np.zeros((367,2048))
    pop = 0
    for data in fake_loader:
        data_act = tezhen(data)
        start = pop*batch_size
        end = (pop+1)*batch_size
        if end >367:
            end = 367
        data_act= data_act.detach().numpy()
        act1[start:end] = data_act
        pop +=1
    pop=0
    for data in truth_loader:
        data_act = tezhen(data)
        start = pop*batch_size
        end = (pop+1)*batch_size
        if end >367:
            end = 367
        data_act= data_act.detach().numpy()
        act2[start:end] = data_act
        pop +=1
    print(cal_kid(act1,act2))
    print(cal_fid(act1,act2))