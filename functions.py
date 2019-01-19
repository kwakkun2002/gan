import os
from PIL import Image

def rename(dir_name):
    os.chdir("{}".format(dir_name))
    num = 0
    for filename in os.listdir("."):
        file_type = filename.split('.')[1]
        if (file_type == '.JPEG' or file_type == '.JPG'):
            os.rename(filename, str(num) + '.jpeg')
        elif (file_type == '.PNG'):
            os.rename(filename, str(num) + '.png')
        num+=1


def imgSizeChanger(size,file_name,new_file_name):
    new_img = Image.new("RGB", size, "white")
    im = Image.open(file_name)
    im.thumbnail(size, Image.ANTIALIAS)
    load_img = im.load()
    load_newimg = new_img.load()
    i_offset = (256 - im.size[0]) / 2
    j_offset = (256 - im.size[1]) / 2
    for i in range(0, im.size[0]):
        for j in range(0, im.size[1]):
            load_newimg[i + i_offset,j + j_offset] = load_img[i,j]
    new_img.save(new_file_name, "JPEG")

for i in range(200):
    imgSizeChanger((256,256),'face/{}__face.jpg'.format(i),'face/{}__face.jpg'.format(i))