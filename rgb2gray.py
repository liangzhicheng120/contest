#coding:utf-8
from PIL import Image
from PIL import Image as image
import numpy as np
import os

def RGB2Gray(image_origin):
    gray = image_origin.convert('L')
    return gray

def saveImage(image_final,i):
    image_final.save("{0}_gray.png".format(i))

#二值化
def binaryzation(image_gray):
    image_gray  = image_gray.resize((180,60),image.ANTIALIAS)
    image_array = np.array(image_gray)
    image_array = (image_array/255.0)*(image_array/255.0)
    #image_array = 1-np.rint(image_array) 
    image_array = 1-image_array
    image_array = 1+np.floor(image_array - image_array.mean())
    return image_array


if __name__ == "__main__":
    for i in range(100000):
        os.system("clear")
        print(i)
        image_handle = Image.open("../image_contest_level_1/{0}.png".format(i))
        image_handle = RGB2Gray(image_handle)
        image_array = binaryzation(image_handle)
        image_array = 255.0*(1-image_array)
        image_handle = Image.fromarray(np.uint8(image_array))
        saveImage(image_handle,i)
