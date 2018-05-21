# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 22:44:17 2017

@author: FUNNICLOWN
"""

import pygame
import sys
from pygame.locals import *

def save_font(all_font, all_text, screen):
    "全ての画像表示"
    font_no = len(all_font)
    text_no = len(all_text)
    
    magni = 80
    for i in range(text_no):
        for j in range(font_no):
            sysfont = pygame.font.SysFont(all_font[j], magni)
            text = sysfont.render(all_text[i], True, (0, 0, 0))
            screen.fill((255, 255, 255))            
            screen.blit(text, (0, 0))
            pygame.display.update()
#            pygame.display.set_mode()
    
            "フォントの画像を保存"
            file_name = all_text[i] + "(" + all_font[j] + ")" + ".png"
            pygame.image.save(screen, file_name)
    
    
"画面設定"
screen_size = [100, 100]
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("font")

"フォント設定"
all_font = ["bodonipostercompressed", "broadway", "dejavusans", "dejavuserif",
            "gentiumbasic", "gentiumbookbasic", "gillsansultra", "myanmartext", "rockwellextra", "tahoma"]
all_text = "/"

"描画"
count = 0
while True:
    screen.fill((255, 255, 255))
    
    save_font(all_font, all_text, screen)
    
    "1回実行したら終了"
    if count == 0:
        pygame.quit()
        sys.exit()
    count == 1
    
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit()

#raw_input("end program")