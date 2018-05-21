# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:34:36 2017

@author: 7317058
"""

"button"

import pygame
from pygame.locals import *
import sys

from text_painting import Text_painting

class Button():
    def __init__(self, screen, text_pic, button_pos=[0, 0], button_size=[100, 50], button_color=(0,80,0), font="broadway", button_name="button", button_id="button"):
        self.screen = screen        
        self.button_pos = button_pos
        self.button_size = button_size
        self.button_name = button_name
        self.button_color = button_color
        self.text_pic = text_pic
        self.button_id = button_id
        
    def shape(self):
        "外枠描画"
        pygame.draw.ellipse(self.screen, self.button_color, Rect(self.button_pos[0], self.button_pos[1], self.button_size[0], self.button_size[1]))
        #今のままでは画像と被り表示されない。
        
        "ボタン名描画"
        Button_name = Text_painting()
        Button_name.text_paint(self.screen, self.text_pic, self.button_pos, self.button_name, self.button_size)
        
    def action(self):
        global button_down, down_count
        
        result = False
        
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()        
        
        if mouse_pressed[0] and down_count == 0:
            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                button_down = [1, self.button_id]
                down_count += 1
                
        if button_down == [1, self.button_id]:
            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                if mouse_pressed[0] == 0:
                    result = True
                    
                    "count reset"
                    down_count = 0
                    button_down = 0
                    
        "count reset"
        if mouse_pressed[0] == 0:
            down_count = 0
            if self.button_pos[0] > mouse_pos[0] > self.button_pos[0]+self.button_size[0] or self.button_pos[1] > mouse_pos[1] > self.button_pos[1]+self.button_size[1]:
                button_down = 0
                
        return result