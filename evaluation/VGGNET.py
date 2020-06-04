import torch.nn as nn
from torchvision import models



class VGGNET(nn.Module):
    def __init__(self, num_classes=1024):	   #num_classes，此处为 二分类值为2
        super(VGGNET, self).__init__()
        net = models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()	#将分类层置空，下面将改变我们的分类层
        self.features = net		#保留VGG16的特征层
        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 4096),  #512 * 7 * 7不能改变 ，由VGG16网络决定的
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
