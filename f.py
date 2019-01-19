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

mb_size = 64    # 미니배치 사이즈는 64

transform = transforms.Compose([transforms.ToTensor()])     # 들어온 이미지를 텐서로 바꿈

train_set = torchvision.datasets.MNIST(root='./NewData',download=False,train=True,transform=transform)  # mnist데이터 셋을 가져옴

trainloader = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=mb_size)    # 가져온 데이터 셋을 미니배치사이즈로 자름

data_iter = iter(trainloader)   #trainloader를 iterable하게 바꿈

images,labels = data_iter.next()    #호출 후 값을 images와 labels에 저장, image는 진짜 숫자이미지 labels는 숫자들 이거 64개가 한 세트이다

test = images.view(images.size(0),-1) # 이미지를 평평하게 만듬

Z_dim = 100 # Z_dim을 만들고 100으로 설정
X_dim = test.size(1) # X_dim 은 여기서는 784

def imgshow(img): #그냥 이미지 출력하는 함수
    im = torchvision.utils.make_grid(img)
    npimg = im.numpy()
    print(npimg.shape)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()

h_dim = 50 #어디에 쓰는지

def init_weight(m): #weight초기화 하는 함수, 선형이라면 샤비에 이니셜라이져를 씀
    if type(m) is nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0) #바이오스는 0으로 초기화

class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__() #초기화
        self.model = nn.Sequential( #케라스와 비슷하다
            nn.Linear(Z_dim,h_dim), #100->50
            nn.ReLU(),
            nn.Linear(h_dim,X_dim), #50-> 784
            nn.Sigmoid()
        )
        self.model.apply(init_weight) # 모델안에 초기화 함수를 적용
    def forward(self, input):
        return self.model(input)



class Dis(nn.Module):
    def __init__(self):
        super(Dis,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim,h_dim), #784 -> 50
            nn.ReLU(),
            nn.Linear(h_dim,1), #50->1
            nn.Sigmoid()
        )
        self.model.apply(init_weight)
    def forward(self, input):   #forward 함수
        return self.model(input)

G = Gen()   # 객체 생성
D = Dis()

G_solver = optim.Adam(G.parameters(),lr = 1e-3)     #옵티마이저 설정
D_solver = optim.Adam(D.parameters(),lr = 1e-3)


for epoch in range(2):
    for i,data in enumerate(trainloader): # trainloader에는 mnist가 들어있는데, data에 그것을 계속 보내준다 i에는 걍 숫자가 있다
        X ,_ = data # X에는 숫자의 배열(mnist의 그것)이 들어가고 _에는 인덱스가 있다.

        mb_size = X.size(0)     #64를 가짐
        X = X.view(X.size(0),-1)    #그걸 평평하게 함 28*28 이 784가 됨

        one_labels = torch.ones(mb_size,1)
        zero_labels = torch.zeros(mb_size,1)

        z = torch.randn(mb_size,Z_dim)  # 64*100의 크기 ,노이즈임
        G_sample = G(z)     #샘플(노이즈)를 받는다
        D_fake = D(G_sample)    #D에게 이 인풋(노이즈)가 가짜라고 설정한다.
        D_real = D(X)   #D가 진짜인걸 넣는다

        D_fake_loss = F.binary_cross_entropy(D_fake,zero_labels)    #가짜에 대한 loss는 0과 비교
        D_real_loss = F.binary_cross_entropy(D_real,one_labels)     #진짜에 대한 loss는 1과 비교(진짜는 1이 되어야 하기때문)
        D_loss = D_fake_loss + D_real_loss      #D의 loss는 가짜에 대한 로스와 진짜에 대한 로스를 더한 값이다. 최대한 가짜는 가짜라고 말하고
        # 진짜는 진짜라고 말하는 것을 학습

        D_solver.zero_grad()    #역전파전에 그라디언트를 0으로
        D_loss.backward()   #역전파 시작
        D_solver.step()     # 변수(그라디언트)를 업데이트

        z = torch.randn(mb_size,Z_dim) #64*100크기의 렌덤, 노이즈임
        G_sample = G(z)     #다시 가짜 노이즈를 만듬
        D_fake = D(G_sample) #가짜를 가짜라고 설정
        G_loss = F.binary_cross_entropy(D_fake,one_labels)      # G의 loss는 G가 넣은 값에 D가 판단한 값이 1이 되게 해야한다

        G_solver.zero_grad()    #G의 변수를 0으로 설정
        G_loss.backward()   #역전파
        G_solver.step()     # 업데이트

        print('Epoch:{},        G_loss:{},      D_loss:{}'.format(epoch,G_loss/(i+1),D_loss/(i+1)))
        samples = G(z).detach()
        samples = samples.view(mb_size,1,28,28)
        answer = samples

imgshow(answer)




















