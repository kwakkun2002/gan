# pytorch 임포트
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
import os
import PIL as Image


def rename(dir_name):
    os.chdir("{}".format(dir_name))
    num = 0
    print(os.listdir("./"))
    for filename in os.listdir("."):
        file_type = filename.split('.')[1]
        # if (file_type == '.JPEG' or file_type == '.JPG'):
        #         #     os.rename(filename, str(num) + '.jpeg')
        #         # elif (file_type == '.PNG'):
        #         #     os.rename(filename, str(num) + '.png')
        os.rename(filename,str(num)+'.jpg')
        num+=1


def imgSizeChanger(size,file_name,new_file_name):
    new_img = Image.new("RGB", size, "white")
    im = Image.open(file_name)
    im.thumbnail(size, Image.ANTIALIAS)
    load_img = im.load()
    load_newimg = new_img.load()
    i_offset = (size - im.size[0]) / 2
    j_offset = (size - im.size[1]) / 2
    for i in range(0, im.size[0]):
        for j in range(0, im.size[1]):
            load_newimg[i + i_offset,j + j_offset] = load_img[i,j]
    new_img.save(new_file_name, "JPEG")

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
#     imgSizeChanger((256,256),'face/{}__face.jpg'.format(i),'face/{}__face.jpg'.format(i))

