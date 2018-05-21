# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:26:12 2017

@author: FUNNICLOWN
"""

import numpy as np
from PIL import Image

def savez_image_data(**kwargs):
    
    option = {"file_name":"untitiled.npz"}
    option.update(kwargs)
    
    file_name = option["file_name"]
    all_fonts = option["all_fonts"]
    all_texts = option["all_texts"]
    no_fonts = len(all_fonts)
    no_texts = len(all_texts)
    no_color_kinds = 3
    
    img = Image.open(all_texts[0] + "(" + all_fonts[0] + ")" + ".png")
    no_hori_pixels, no_ver_pixels = img.size    
    
    n = len(all_texts)
    m = len(all_texts) * len(all_fonts)
    yt = np.zeros((m, n))
    xt = np.zeros((no_fonts, no_texts, no_hori_pixels, no_ver_pixels, no_color_kinds))
    
    img_count = 0
    for i in range(no_fonts):
        for j in range(no_texts):
            img = Image.open(all_texts[j]+"("+all_fonts[i]+").png")
            yt[img_count, j] = 1
            img_count += 1
            
            for k in range(no_ver_pixels):
                for l in range(no_hori_pixels):
                    xt[i, j, k, l] = img.getpixel((k, l))
                    
    print(xt)
    print(yt)
    np.savez(file_name, x=xt, y=yt)
    
all_fonts = ["bodonipostercompressed", "broadway", "dejavusans", "dejavuserif",
             "gentiumbasic", "gentiumbookbasic", "gillsansultra", "myanmartext", "rockwellextra", "tahoma"]
all_texts = "abcdefghijklmnopqrstuvwxyz"

savez_image_data(all_fonts=all_fonts, all_texts=all_texts, file_name="alphabet.npz")
