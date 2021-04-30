import cv2
import time
import numpy as np

class ImageProcessing:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.image 
        time.sleep(2)
        self.red_mask_low
        self.red_mask_up
        self.yellow_mask_low
        self.yellow_mask_up
    
    def capture(self):
        ret,frame = self.cap.read()
        self.image = frame

    def calibration(self) :
        self.capture()
        frame = self.image
        raw_frame = cv2.flip(frame,1)
        hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        #pos_w = [60,250,430,610]
        pos_w = [60,610]
        pos_h = [55,150,230,310,390,460]

        point_num = 5
        
        red_h_list = []
        red_s_list = []
        red_v_list = []

        yellow_h_list = []
        yellow_s_list = []
        yellow_v_list = []

        for index_h in range(len(pos_h)) :
            for index_w in range(len(pos_w)) :

                sum_h = []
                sum_s = []
                sum_v = []
                for index_point in range(point_num) :
                    
                    if (index_h+index_w) % 2 == 0 : # red point
                        sum_h.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                        sum_s.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                        sum_v.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
                    else : # yellow point
                        sum_h.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                        sum_s.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                        sum_v.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
                        
                    #cv2.circle(copy_image, (pos_w[index_w],pos_h[index_h]-index_point) , 3 , (255,255,255), -1)
                
        
                if (index_h+index_w) % 2 == 0 : # red token
                    print('red')
                    red_h_list.append( sum(sum_h)/point_num )
                    red_s_list.append( sum(sum_s)/point_num )
                    red_v_list.append( sum(sum_v)/point_num )
                else : # yellow token
                    print('yellow')
                    yellow_h_list.append( sum(sum_h)/point_num )
                    yellow_s_list.append( sum(sum_s)/point_num )
                    yellow_v_list.append( sum(sum_v)/point_num )

        epsilon = 5

        red_h_low = min(red_h_list) - epsilon
        red_s_low = min(red_s_list) - epsilon
        red_v_low = min(red_v_list) - epsilon

        red_h_up = max(red_h_list) + epsilon
        red_s_up = max(red_s_list) + epsilon
        red_v_up = max(red_v_list) + epsilon

        yellow_b_low = min(yellow_h_list) - epsilon
        yellow_g_low = min(yellow_s_list) - epsilon
        yellow_r_low = min(yellow_v_list) - epsilon

        yellow_b_up = max(yellow_h_list) + epsilon
        yellow_g_up = max(yellow_s_list) + epsilon
        yellow_r_up = max(yellow_v_list) + epsilon

        self.red_mask_low = [red_h_low,red_s_low,red_v_low]
        self.red_mask_up = [red_h_up,red_s_up,red_v_up]

        self.yellow_mask_low = [yellow_b_low,yellow_g_low,yellow_r_low]
        self.yellow_mask_up = [yellow_b_up,yellow_g_up,yellow_r_up]

    
    def process_image(self):
        self.capture()
        frame = self.image
        board = np.zeros((6,7),dtype=np.int8)
        raw_frame = cv2.flip(frame,1)
        hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)

        pos_w = [60,150,250,340,430,520,610]
        pos_h = [55,150,230,310,390,460]

        point_num = 5

        for index_h in range(len(pos_h)) :
            for index_w in range(len(pos_w)) :
            
                sum_h = []
                sum_s = []
                sum_v = []
            
                for index_point in range(point_num) :
                    sum_h.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                    sum_s.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                    sum_v.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
                
                ch_h = sum(sum_h)/point_num
                ch_s = sum(sum_s)/point_num
                ch_v = sum(sum_v)/point_num
                            
                if (ch_h >= self.red_mask_low[0] and ch_h <= self.red_mask_up[0]) and (ch_s >= self.red_mask_low[1] and ch_s <= self.red_mask_up[1]) and (ch_v >= self.red_mask_low[2] and ch_v <= self.red_mask_up[2]) : 
                    board[index_h][index_w] = 2 # red detected = 2
                
                elif (ch_h >= self.yellow_mask_low[0] and ch_h <= self.yellow_mask_up[0]) and (ch_s >= self.yellow_mask_low[1] and ch_s <= self.yellow_mask_up[1]) and (ch_v >= self.yellow_mask_low[2] and ch_v <= self.yellow_mask_up[2]) : 
                    board[index_h][index_w] = 1 # yellow detected = 1
        
        return board