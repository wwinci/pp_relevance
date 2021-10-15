import cv2
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from get_feature.model_resnet import ft_net

class Cal_features():
    def __init__(self, method='reid'):
        """利用get_features函数，输入图像，输出特征

        Args:
            method (str, optional): 选择提取特征方法， 可选“vgg”-vgg网络, 
                                                    "hist"-直方图匹配
                                                    "both"-二者都用，特征cat起来. 
                                                    "reid"-reid预训练的resnet模型
                                    Defaults to 'hist'. 
        """
        self.method = method
        if self.method == 'vgg' or self.method == 'both':
            self.model = models.vgg16(pretrained=True).cuda()
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

        elif self.method == 'reid':

            self.model = ft_net()# .cuda()
            weights = torch.load('./get_feature/net_last.pth')
            pop_key = []
            for key in weights.keys():
                if 'classifier.classifier' in key or 'classifier.add_block.1' in key:
                    pop_key.append(key)
            for key in pop_key:
                weights.pop(key)
            self.model.load_state_dict(weights)
        # elif self.method == 'hist':
        #     self.bn = nn.BatchNorm1d(3*256)

    def get_features(self, image):
        features = []
        if self.method == 'vgg' or self.method == 'both':
            self.model.eval()
            feature_vgg = self.model.features(image)
            # print(feature_vgg.shape)
            features.append(self.pool(feature_vgg))
        if self.method == 'hist' or self.method == 'both':
            n, c, h, w = image.size()
            image *= image*255
            image = image.long().view(c, n*h*w)
            for channel in range(c):
                feature_hist = torch.histc(image[channel], bins=256, min=0, max=255).view((1, -1, 1, 1)).float()
                feature_hist[0, 0] = 0
                feature_hist = F.normalize(feature_hist, p=1, dim=1)
                features.append(feature_hist)
        if self.method == 'reid':
            self.model.eval()
            with torch.no_grad():
                feature_reid = self.model(image)
            features.append(feature_reid)
            # print(self.method)

        features = torch.cat(features, dim=1)
            
        # print(features.shape)
        # print(torch.max(features))
        return features

if __name__ == "__main__":
    # inp = 'tmp/img/backpack0_1.png'
    # outp = 'tmp/img/backpack1_1.png'
    inp = 'tmp/1.jpg'
    outp = 'tmp/2.jpg'
    a = cv2.imread(inp)
    a = torch.from_numpy(a).float() # cuda
    a = a / 255.0
    h, w, _ = a.shape
    a = a.permute(2, 0, 1).contiguous().unsqueeze(0)
    print(a.shape, type(a))

    b = cv2.imread(outp)
    b = torch.from_numpy(b).float() # cuda
    b = b / 255.0
    h, w, _ = b.shape
    b = b.permute(2, 0, 1).contiguous().unsqueeze(0)
    print(b.shape, type(b))

    cal = Cal_features(method='reid')
    feature_a = cal.get_features(a)
    feature_b = cal.get_features(b)

    print('feature shape: ', feature_a.shape, feature_b.shape)

    print(torch.cosine_similarity(feature_a, feature_b))