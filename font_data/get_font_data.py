# coding: cp932
"""
Created on Tue Oct 31 12:07:16 2017

@author: 7317058
"""
from PIL import Image
import numpy as np

"�f�[�^�t�@�C���쐬"

def get_font_data(all_fonts, all_text):
    """pixels�擾"""
    m = len(all_fonts)
    n = len(all_text)
#    "test"
#    m = 7
#    n = 13
        
    "img_pixels�͑S�Ă�pixels���擾�B"
    "�ŏI�I�Ȍ^�́A(height * width) * 3(RGB)��(m * n)������^�ɂȂ�"
    img_pixels = [0 for i in range(m)]
    
    "�摜�t�@�C�����J�� & �s�N�Z���̐F���擾"
    for i in range(m):
        al_pixels = []
        for j in range(n):
            "�T�C�Y�擾"
            img = Image.open(all_text[j]+"("+all_fonts[i]+").png")
            width, height = img.size
            
#            "test"
#            width = 73
#            height = 100
            
            
            "pixels�擾"
            y_pixels = []
            for y in range(height):
                x_pixels = []
                for x in range(width):
                    x_pixels.append(img.getpixel((x, y)))
                
                "�摜�P����pixels�擾"
                y_pixels.append(x_pixels)
                
            "������1��ނ�pixels�擾"
            al_pixels.append(y_pixels)
            
        "�S�Ă̕�����pixels�擾"
        al_pixels = np.array(al_pixels)
        img_pixels[i] = al_pixels
                                
    "�摜���"
    print("������     : " + str(n) + " ����")
    print("�t�H���g��    : " + str(m) + " ���")
    print("�摜����   : " + str(m * n) + " ��")
    print("�� pixel�� : "+str(width))
    print("�c pixel�� : "+str(height))
    print("np.shape(img_pixels) = " + str(np.shape(img_pixels)))

#    "check"
#    print(y_pixels)
#    print(np.shape(y_pixels))
    return img_pixels

def write_in_file(img_pixels, file_name="untitled.npy"):
    "�t�@�C���ւ�pixel���̏�������"
    np.save(file_name, img_pixels)
    
"test���R�����g�����Ďg������"
#"alphabet.npy"
#all_fonts = ["bodonipostercompressed", "broadway", "dejavusans", "dejavuserif",
#             "gentiumbasic", "gentiumbookbasic", "gillsansultra", "myanmartext", "rockwellextra", "tahoma"]

"alphabet2.npy"
all_fonts = ["dejavusans", "dejavuserif", "gentiumbasic", "gentiumbookbasic", "myanmartext"]

             
all_text = "abcdefghijklmnopqrstuvwxyz"

img_pixels = get_font_data(all_fonts, all_text)
#print(img_pixels)

"�ۑ�"
file_name = "alphabet2.npy"
write_in_file(img_pixels, file_name)

check = np.load(file_name)
print("�t�@�C���f�[�^�m�F(np.shape) : " + str(np.shape(check)))
raw_input("check")

