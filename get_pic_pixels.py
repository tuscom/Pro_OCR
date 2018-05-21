# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:38:45 2017

@author: FUNNICLOWN
"""

from PIL import Image
import numpy as np
import time
#import numba

#@numba.jit
def get_pic_pixelR(file_name, rect=["x","y","width","height"], resize=["re_width", "re_height"]):
    start = time.time()
    "渡された画像から、指定された範囲のpixelを取得し、返す"
    "今回はR成分のみを返す"
    img = Image.open(file_name)
    
    pixel_storeR = np.zeros((resize[0], resize[1]))
    pixel_for_resize = np.zeros((rect[2], rect[3], 3))
    pixel_for_resize = pixel_for_resize.astype(np.int64)
    
    img_abstract = Image.new("RGB", (rect[2], rect[3]))
    for x in range(rect[2]):
        for y in range(rect[3]):
            pixel_for_resize[x, y, :] = img.getpixel((x+rect[0], y+rect[1]))
            img_abstract.putpixel((x, y), 
                                (pixel_for_resize[x, y, 0], pixel_for_resize[x, y, 1], pixel_for_resize[x, y, 2]))
    
    img_resize = img_abstract.resize((resize[0], resize[1]))
    for x in range(resize[0]):
        for y in range(resize[1]):
            pixel_storeR[x, y] = img_resize.getpixel((x, y))[0]
    
    elapsed_time = time.time() - start
    print("get_pic_pixelR's time : " + str(elapsed_time))
    
    return pixel_storeR #=(re_width, re_height)
    
#"test"
#file_name = "set_font_data.png"
#rect = [30, 180, 500, 500]
#resize = [100, 100]
#pixelR = get_pic_pixelR(file_name, rect, resize)
#
#test_img = Image.new("RGB", (rect[2], rect[3]))
#for x in range(rect[2]):
#    for y in range(rect[3]):
#        test_img.putpixel((x, y), (int(pixelR[x, y]), int(pixelR[x, y]), int(pixelR[x, y])))
#
#test_img.show()