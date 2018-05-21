# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:27:11 2017

@author: 7317058
"""

"pro_OCR"

import pygame
from pygame.locals import *
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

from text_painting import Text_painting
from get_pic_pixels import get_pic_pixelR
from accuracy import predict
from key_information import key_info
from theta_modules import update_theta, a0_to_yp
from J_modules import calculate_J
#from button import Button
#globalはファイルを渡れないので、同じファイル内にモジュールを書くことになった

class Button():
    def __init__(self, screen, text_pic="", button_pos=[0, 0], button_size=[100, 50], button_color=(0,80,0), font="broadway", button_name="button", button_id="button"):
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
        
        if mouse_pressed[0] and down_count[self.button_id] == 0:
            down_count[self.button_id] += 1

            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:

                button_down[self.button_id] = [1, self.button_id]
                
        if button_down[self.button_id] == [1, self.button_id]:
            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                if mouse_pressed[0] == 0:
                    result = True
                    
                    "count reset"
                    down_count[self.button_id] = 0
                    button_down[self.button_id] = 0
                    
        "count reset"
        if mouse_pressed[0] == 0:
            down_count[self.button_id] = 0
#            if self.button_pos[0] > mouse_pos[0] > self.button_pos[0]+self.button_size[0] or self.button_pos[1] > mouse_pos[1] > self.button_pos[1]+self.button_size[1]:
            if not(self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]):
                button_down[self.button_id] = 0
                
        return result
        
    def action_long_true(self):
        global button_down, down_count

        result = False
        
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()        
        
        if mouse_pressed[0] and down_count[self.button_id] == 0:
            down_count[self.button_id] += 1

            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                button_down[self.button_id] = [1, self.button_id]
                
        if button_down[self.button_id] == [1, self.button_id]:
            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                result = True
                
        "count reset"
        if mouse_pressed[0] == 0 or not(self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]):
            down_count[self.button_id] = 0
            button_down[self.button_id] = 0
                
        return result
        
    def only_down(self):
        
        global button_down, down_count

        result = False
        
        mouse_pressed = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()        

        if mouse_pressed[0] and down_count[self.button_id] == 0:
            down_count[self.button_id] += 1

            if self.button_pos[0] <= mouse_pos[0] <= self.button_pos[0]+self.button_size[0] and self.button_pos[1] <= mouse_pos[1] <= self.button_pos[1]+self.button_size[1]:
                result = True                
                
        "count reset"
        if mouse_pressed[0] == 0:
            down_count[self.button_id] = 0
            button_down[self.button_id] = 0
                
        return result


def draw_lines(screen, frame_pos, frame_size, line_color=(0, 0, 0), line_thick=40):

    "お絵描き機能"
    global mouse_pos_store
    
    mouse_pressed = pygame.mouse.get_pressed()
    mouse_pos = pygame.mouse.get_pos()
    
    draw_button = Button(screen, button_pos=frame_pos, button_size=frame_size,button_id="draw_button")
    draw_down_button = Button(screen, button_pos=frame_pos, button_size=frame_size,button_id="draw_down_button")    

    if draw_down_button.only_down():
        mouse_pos_store.append([]) #次のが用意される

    only_down_count = len(mouse_pos_store) - 1


    if draw_button.action_long_true():
        mouse_pos_store[only_down_count-1].append(mouse_pos)
    
    "取得した座標をlinesとして描画"
    if only_down_count > 0:
        for i in range(len(mouse_pos_store)-1): #次の座標を入れるからの型をforループに含まれないようにする
            if len(mouse_pos_store[i]) > 1:
                pygame.draw.lines(screen, line_color, False, mouse_pos_store[i], line_thick)


"画面設定"
pygame.init()
window_size = [1500, 800]
page_color = (200, 200, 200)
screen = pygame.display.set_mode((window_size[0], window_size[1]))
pygame.display.set_caption("pro OCR")

"pageフェイズ"
phase = [1, 1, 1]

"文字描画用画像ロード"
Text_paint = Text_painting()
text_pic = Text_paint.text_load()

"実行回数カウント"
do_file_name = "do_count.npz"
do_file = np.load(do_file_name)
action_count = do_file["action_count"]

"ボタン制御用変数"
id_list = ["decide_button", "erase_button", "start_button", "draw_button", 
           "draw_down_button", "hit_button", "miss_button", "end_button",
           "again_button", "correct_button", "back_button", "real_time_button"]
id_init = [0 for i in range(len(id_list))]
down_count = dict(zip(id_list, id_init))
button_down = dict(zip(id_list, id_init))

"お絵描き部分制御用変数"
mouse_pos_store = [[]]
frame_rect = [30, 180, 500, 500]
frame_pos = [frame_rect[0], frame_rect[1]]
frame_size = [frame_rect[2], frame_rect[3]]
frame_resize = [100, 100]
label_text = "abcdefghijklmnopqrstuvwxyz" #予測できる文字の種類

"グラフ描画用変数"
accuracy_graph_name = "accuracy_change.png"
accuracy_file_name = "accuracy.npz"
accuracy_graph = pygame.image.load(accuracy_graph_name)
accuracy_file = np.load(accuracy_file_name)
update_store = accuracy_file["update_store"]
accuracy_store = accuracy_file["accuracy_store"]
random_select = 10
j_ite_graph_name = "j_ite.png"
j_ite_graph = pygame.image.load(j_ite_graph_name)
j_lambda_graph_name = "j_lambda.png"
j_lambda_graph = pygame.image.load(j_lambda_graph_name)

"リアルタイム描画用変数"
file_real_time = "real_time_data.png"
update_count_for_real_time = 0
predict_real_time_label = " "

"input_bar用変数"
bar_display_count = 0
input_text = " "

"操作可能パラメータ"
update_ite = 100
lam = 0.01
alpha = 0.001
"phase5で追加表示するもの"

"必要なデータロード"
theta_file_name = "theta.npz"
theta_file = np.load(theta_file_name)
theta = theta_file["theta"]
train_data_file_name = "alphabet.npz"
train_data_file = np.load(train_data_file_name)
a0 = train_data_file["a0"]
yt = train_data_file["yt"]

while True:
    screen.fill(page_color)
    
    if phase[0] == 1:
        "スタート画面"
        Text_paint.text_paint(screen, text_pic, pos=[window_size[0]/2-250, 100], text="mini ocr", size=[500, 300])
        start_button = Button(screen, text_pic, button_pos=[window_size[0]/2-150, window_size[1]-300], button_size=[300, 150], button_name="start", button_id="start_button")
        start_button.shape()
        
        if start_button.action():
            phase[0] = 2
            action_count += 1
#            np.savez(do_file_name, action_count=action_count)
            
    if phase[0] == 2:
        "count描画"
        action_text = "action count " + str(action_count) + " times"
        update_text = "update count " + str(update_store[-1]) + " times"
        Text_paint.text_paint(screen, text_pic, pos=[30, 0], text=action_text, size=[500, 80])
        Text_paint.text_paint(screen, text_pic, pos=[30, 80], text=update_text, size=[500, 80])
    
        "枠描画"
        pygame.draw.rect(screen, (255, 255, 255), Rect(frame_pos[0], frame_pos[1], frame_size[0], frame_size[1]))
        "お絵描き部分"
        draw_lines(screen, frame_pos, frame_size)
        
        "accuracyグラフ描画"        
        screen.blit(accuracy_graph, (1000, 200))
        "Jの描画"
        screen.blit(j_ite_graph, (600, 500))
        screen.blit(j_lambda_graph, (1000, 500))
        #グラフのサイズは大体(400, 300)

        
        "決定ボタン"
        decide_button = Button(screen, text_pic, button_pos=[30, 700], button_size=[200, 100], button_name="set", button_id="decide_button")
        decide_button.shape()
        if decide_button.action():
           print("set")
           phase[0] = 3
           mouse_pos_store = [[]]
           predict_real_time_label = " "

           "area内のpixelR情報取得"
           file_final_draw = "set_font_data.png"
           pygame.image.save(screen, file_final_draw) #画像保存（上書き）
           pixel_area_final = get_pic_pixelR(file_final_draw, rect=frame_rect, resize=frame_resize) #pixel取得

#           pixel_area_final = np.reshape(pixel_area_final, (frame_resize[0] * frame_resize[1])) #計算用に型変換
           predict_final_label = predict(theta, pixel_area_final, label_text)#label推測
            
        "やり直しボタン"
        erase_button = Button(screen, text_pic, button_pos=[300, 700], button_size=[200, 100], button_name="erase", button_id="erase_button")
        erase_button.shape()
        if erase_button.action():
            mouse_pos_store = [[]]
            
        
        "リアルタイム予測結果の描画"
        predict_pos = [600, 30]
        #予測はbutton.action時の方がストレス少なくていいと思う
        Text_paint.text_paint(screen, text_pic, pos=predict_pos, text="prediction", size=[250, 100])
        real_time_button = Button(screen, text_pic, button_pos=frame_pos, button_size=frame_size, button_name=" ", button_id="real_time_button")
        if real_time_button.action():
            pygame.image.save(screen, file_real_time)
            pixel_real_time = get_pic_pixelR(file_real_time, rect=frame_rect, resize=frame_resize)
            predict_real_time_label = predict(theta, pixel_real_time, label_text)#label推測
#        update_count_for_real_time +=1
#        if update_count_for_real_time % 1000 == 0: #部分的な更新間隔制御
#            pygame.image.save(screen, file_real_time)
#            pixel_real_time = get_pic_pixelR(file_real_time, rect=frame_rect, resize=frame_resize)
#            predict_real_time_label = predict(theta, pixel_real_time, label_text)#label推測
        Text_paint.text_paint(screen, text_pic, pos=[predict_pos[0]+50, predict_pos[1]+120], text=predict_real_time_label, size=[150, 150])
        
    if phase[0] == 3:
        "area内の画像について"
        predict_text = "your answer is"
        Text_paint.text_paint(screen, text_pic, pos=[300, 30], text=predict_text, size=[800, 150])
        Text_paint.text_paint(screen, text_pic, pos=[550, 250], text=predict_final_label, size=[400, 400])        
        
        
        "yes_button"
        hit_button = Button(screen, text_pic, button_pos=[200, 600], button_size=[300, 150], button_name="yes", button_id="hit_button")
        hit_button.shape()
        if hit_button.action():
            print("hit")
            phase[0] = 6
            
        "no_button"
        miss_button = Button(screen, text_pic, button_pos=[1000, 600], button_size=[300, 150], button_name="no", button_id="miss_button")
        miss_button.shape()
        if miss_button.action():
            print("miss")
            phase[0] = 4
            
    if phase[0] == 4:
        "question"
        Text_paint.text_paint(screen, text_pic, pos=[300, 50], text="the answer is", size=[800, 150])
        
        "input_bar"
        bar_rect = [650, 300, 150, 150]
        pygame.draw.rect(screen, (240, 240, 240), Rect(bar_rect[0], bar_rect[1], bar_rect[2], bar_rect[3]))
        
        bar_display_count += 1
        if (bar_display_count / 500) % 2 and input_text == " ":
            pygame.draw.line(screen, (0, 0, 0), bar_rect[:2], [bar_rect[0], bar_rect[1]+bar_rect[3]], 10)
        
        input_text = key_info(input_text)
        Text_paint.text_paint(screen, text_pic, pos=bar_rect[:2], text=input_text, size=bar_rect[2:])
        
        "決定ボタン"
        correct_button = Button(screen, text_pic, button_pos=[300, 500], button_size=[200, 100], button_name="ok", button_id="correct_button")
        correct_button.shape()
        if input_text != " " and correct_button.action():
            print("correct")
            phase[0] = 5
            
        "戻るボタン"
        back_button = Button(screen, text_pic, button_pos=[1000, 500], button_size=[200, 100], button_name="back", button_id="back_button")
        back_button.shape()
        if back_button.action():
            print("back")
            phase[0] = 3
            input_text = " "
            
    if phase[0] == 5:
        screen.fill(page_color)
        start = time.time()
        "描画部分"
        Text_paint.text_paint(screen, text_pic, pos=[100, 30], text="now updating...", size=[400, 100])
        Text_paint.text_paint(screen, text_pic, pos=[80, 300], text="update_iteration = "+str(update_ite), size=[800, 80])
        Text_paint.text_paint(screen, text_pic, pos=[80, 380], text="lambda = "+str(lam), size=[400, 80])
        Text_paint.text_paint(screen, text_pic, pos=[80, 460], text="alpha = "+str(alpha), size=[400, 80])
        Text_paint.text_paint(screen, text_pic, pos=[80, 540], text="random_select = "+str(random_select), size=[500, 80])
        Text_paint.text_paint(screen, text_pic, pos=[80, 620], text="boundary = linear", size = [600, 80])
        pygame.display.update()
        
        "処理部分"
        #長いから進行度を表示する
        "訓練データ追加"
        a0_add = pixel_area_final.flatten()
        a0_add = np.r_[1, a0_add]
        a0_add = np.reshape(a0_add, (1, np.shape(a0_add)[0]))
        a0 = np.r_[a0, a0_add]
        
        #ytはinput_textをラベル変換したもの
        yt_add = np.zeros((1, len(label_text)))
        yt_add[0, label_text.index(input_text)] = 1
        yt = np.r_[yt, yt_add]
        input_text = " "
        
#        np.savez(train_data_file_name, a0=a0, yt=yt)
        
        "θ更新"
        theta = update_theta(theta, a0, yt, lam, alpha, update_ite, theta_file_name)
        
        "accuracy保存"
        yp = a0_to_yp(a0, theta)
    
        m = np.shape(a0)[0]
        bit_hit = yp==yt
        "各アルファベット評価"
        alpha_hit = np.all(bit_hit, axis=1)
        "正当要素抽出"
        hit = float(alpha_hit[alpha_hit == True].size)

        m = float(m)
        ratio = 100. * (hit / m)
        print("精度 : " + str(ratio) + "%")

        update_store = np.append(update_store, update_store[-1]+update_ite)
        accuracy_store = np.append(accuracy_store, ratio)
            
#        np.savez(accuracy_file_name, update_store=update_store, accuracy_store=accuracy_store)
#        np.savez(theta_file_name, theta=theta)

        fig = plt.figure()
        plt.title("accuracy change")
        plt.xlabel("update iteration")
        plt.ylabel("accuracy")
        plt.plot(update_store, accuracy_store)
        plt.savefig(accuracy_graph_name)
        accuracy_graph = pygame.image.load(accuracy_graph_name)

        "Jのグラフ保存"
        calc_J = calculate_J([a0, yt], theta, alpha, update_ite=update_ite, random_select=random_select)
        calc_J.j_ite(lam)
        calc_J.j_lam()
        j_ite_graph = pygame.image.load(j_ite_graph_name)
        j_lambda_graph = pygame.image.load(j_lambda_graph_name)
        
        "フェイズ移行"
        phase[0] = 6
        
        elapsed_time = time.time() - start
        print("phase5's time : " + str(elapsed_time))
                            
    if phase[0] == 6:
        "感謝の言葉"
        Text_paint.text_paint(screen, text_pic, pos=[100, 30], text="thank you for using.", size=[1300, 200])
        
        "終了ボタン"
        end_button = Button(screen, text_pic, button_pos=[300, 300], button_size=[250, 200], button_name="end", button_id="end_button")
        end_button.shape()
        if end_button.action():
            print("end")
            phase[0] = 1
            
        "もう一度ボタン"
        again_button = Button(screen, text_pic, button_pos=[1000, 300], button_size=[250, 200], button_name="again", button_id="again_button")
        again_button.shape()
        if again_button.action():
            print("again")
            phase[0] = 2
            
    for event in pygame.event.get():
        "test"
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
