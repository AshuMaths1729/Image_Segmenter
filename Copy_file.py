import shutil
from glob import glob
import os

src = 'PH2Dataset/PH2_Dataset_images/'
dst = 'Data/'
trail = "*/"

img_folders = [glob(src+"*")]

for i in img_folders[0]:
    imgs_dir = [glob(i + "/*")][0][0:2]
    for j in imgs_dir:
        img = glob(j+"/*")
        for imag in img:
            if 'lesion' in imag:
                imagee = imag[51:][:6] + imag[51:][13:]
                shutil.copy(imag, dst + '/masks/' + imagee)
            else:
                shutil.copy(imag, dst + '/frames/')
                