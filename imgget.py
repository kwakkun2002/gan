import os
from PIL import Image
# # pytorch 임포트
import torch

# 옵티마이저 가져오기
import torch.optim as optim

# 뉴럴네트워크 가져오기
import torch.nn as nn

#  cross_entropy같은거 있음
import torch.nn.functional as F

import torchvision      # 모델 아키텍쳐를 가지고 있음
import torchvision.transforms as transforms
import numpy as np      # 넘파이
import matplotlib.pyplot as plt     #이미지를 보여줄 matplotlib

def imgshow(img): #그냥 이미지 출력하는 함수
    im = torchvision.utils.make_grid(img)
    npimg = im.numpy()
    print(npimg.shape)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


# for i in range(200):
# img = Image.open('./face/{}__face.jpg'.format(0))
# img.resize((2,2))
# img.save('./face/0_changed__face.jpg')
# img = Image.open('./face/0_changed__face.jpg')
# print(img.size)

new_img = Image.new("RGB", (256,256), "white")
im = Image.open('./face/0__face.jpg')
im.thumbnail((256,256), Image.ANTIALIAS)
load_img = im.load()
load_newimg = new_img.load()
i_offset = (256 - im.size[0]) / 2
j_offset = (256 - im.size[1]) / 2
for i in range(0, im.size[0]):
    for j in range(0, im.size[1]):
        load_newimg[i + i_offset,j + j_offset] = load_img[i,j]

new_img.save('./face/0_changed__face.jpg', "JPEG")
# trans = transforms.ToPILImage()
# trans1 = transforms.ToTensor()
# plt.imshow(trans(trans1(img)))
# plt.xticks([])
# plt.yticks([])
# plt.show()











