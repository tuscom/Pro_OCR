# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:40:04 2017

@author: 7317058
"""

"M_1"

import pygame
from pygame.locals import *
import sys, os

class Text_painting():
    def __init__(self):
#        self.all_texts = all_texts
        "nothing action"
        
    def text_load(self, font="broadway"):
        "コードの最初に配置して、描画可能な文字をロードする"
        all_texts = "abcdefghijklmnopqrstuvwxyz0123456789 _/-+.="
        
        text_pic = [0 for i in range(len(all_texts))]
        for i in range(len(all_texts)):

            text_pic[i] = pygame.image.load(os.path.join("font_data", all_texts[i]+"("+font+").png"))
        return text_pic
        
    def text_paint(self, screen, text_pic, pos, text, size, font="broadway"):
        "textに含まれる文字のインデックス化"
        text_index = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9,
                      "k":10, "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, 
                      "u":20, "v":21, "w":22, "x":23, "y":24, "z":25,
                      "0":26, "1":27, "2":28, "3":29, "4":30, "5":31, "6":32, "7":33, "8":34, "9":35,
                      " ":36,
                      "_":37, "/":38, "-":39, "+":40, ".":41, "=":42}
        
        
        one_text_size = [size[0]/len(text), size[1]]#新しいサイズは渡されたサイズ内に収まるようにしなくてはならない。
        for i in range(len(text)):
            blit_text = text_pic[text_index[text[i]]]
            blit_text = pygame.transform.scale(blit_text, one_text_size)
            screen.blit(blit_text, (pos[0] + i * one_text_size[0], pos[1]))
    
#Text = Text_painting()
#Text.text_load()