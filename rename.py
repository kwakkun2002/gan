import os
os.chdir("face")
def rename_file():
    num = 0
    for filename in os.listdir("."):
        os.rename(filename, str(num)+'__face.jpg')
        num+=1
rename_file()