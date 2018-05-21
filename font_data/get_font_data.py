# coding: cp932
"""
Created on Tue Oct 31 12:07:16 2017

@author: 7317058
"""
from PIL import Image
import numpy as np

"データファイル作成"

def get_font_data(all_fonts, all_text):
    """pixels取得"""
    m = len(all_fonts)
    n = len(all_text)
#    "test"
#    m = 7
#    n = 13
        
    "img_pixelsは全てのpixelsを取得。"
    "最終的な型は、(height * width) * 3(RGB)が(m * n)がある型になる"
    img_pixels = [0 for i in range(m)]
    
    "画像ファイルを開く & ピクセルの色を取得"
    for i in range(m):
        al_pixels = []
        for j in range(n):
            "サイズ取得"
            img = Image.open(all_text[j]+"("+all_fonts[i]+").png")
            width, height = img.size
            
#            "test"
#            width = 73
#            height = 100
            
            
            "pixels取得"
            y_pixels = []
            for y in range(height):
                x_pixels = []
                for x in range(width):
                    x_pixels.append(img.getpixel((x, y)))
                
                "画像１枚のpixels取得"
                y_pixels.append(x_pixels)
                
            "文字列1種類のpixels取得"
            al_pixels.append(y_pixels)
            
        "全ての文字のpixels取得"
        al_pixels = np.array(al_pixels)
        img_pixels[i] = al_pixels
                                
    "画像情報"
    print("文字数     : " + str(n) + " 文字")
    print("フォント数    : " + str(m) + " 種類")
    print("画像枚数   : " + str(m * n) + " 枚")
    print("横 pixel数 : "+str(width))
    print("縦 pixel数 : "+str(height))
    print("np.shape(img_pixels) = " + str(np.shape(img_pixels)))

#    "check"
#    print(y_pixels)
#    print(np.shape(y_pixels))
    return img_pixels

def write_in_file(img_pixels, file_name="untitled.npy"):
    "ファイルへのpixel情報の書き込み"
    np.save(file_name, img_pixels)
    
"testをコメント化して使うこと"
#"alphabet.npy"
#all_fonts = ["bodonipostercompressed", "broadway", "dejavusans", "dejavuserif",
#             "gentiumbasic", "gentiumbookbasic", "gillsansultra", "myanmartext", "rockwellextra", "tahoma"]

"alphabet2.npy"
all_fonts = ["dejavusans", "dejavuserif", "gentiumbasic", "gentiumbookbasic", "myanmartext"]

             
all_text = "abcdefghijklmnopqrstuvwxyz"

img_pixels = get_font_data(all_fonts, all_text)
#print(img_pixels)

"保存"
file_name = "alphabet2.npy"
write_in_file(img_pixels, file_name)

check = np.load(file_name)
print("ファイルデータ確認(np.shape) : " + str(np.shape(check)))
raw_input("check")

