# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 05:43:10 2017

@author: FUNNICLOWN
"""

import pygame
from pygame.locals import *

def key_info(input_text):
    pressed_keys = pygame.key.get_pressed()
    
    if pressed_keys[K_a]:
        input_text = "a"
    if pressed_keys[K_b]:
        input_text = "b"
    if pressed_keys[K_c]:
        input_text = "c"
    if pressed_keys[K_d]:
        input_text = "d"
    if pressed_keys[K_e]:
        input_text = "e"
    if pressed_keys[K_f]:
        input_text = "f"
    if pressed_keys[K_g]:
        input_text = "g"
    if pressed_keys[K_h]:
        input_text = "h"
    if pressed_keys[K_i]:
        input_text = "i"
    if pressed_keys[K_j]:
        input_text = "j"
    if pressed_keys[K_k]:
        input_text = "k"
    if pressed_keys[K_l]:
        input_text = "l"
    if pressed_keys[K_m]:
        input_text = "m"        
    if pressed_keys[K_n]:
        input_text = "n"        
    if pressed_keys[K_o]:
        input_text = "o"
    if pressed_keys[K_p]:
        input_text = "p"
    if pressed_keys[K_q]:
        input_text = "q"
    if pressed_keys[K_r]:
        input_text = "r"
    if pressed_keys[K_s]:
        input_text = "s"
    if pressed_keys[K_t]:
        input_text = "t"
    if pressed_keys[K_u]:
        input_text = "u"
    if pressed_keys[K_v]:
        input_text = "v"
    if pressed_keys[K_w]:
        input_text = "w"
    if pressed_keys[K_x]:
        input_text = "x"
    if pressed_keys[K_y]:
        input_text = "y"
    if pressed_keys[K_z]:
        input_text = "z"
    
    return input_text