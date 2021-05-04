import cv2
import time
import numpy as np


def capture() :

    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    
    return frame
    

def calibration(frame) :

    raw_frame = cv2.flip(frame,1)
    hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)

    #pos_w = [60,250,430,610]
    pos_w = [60,610]
    pos_h = [55,150,230,310,390,460]

    point_num = 5
    
    red_b_list = []
    red_g_list = []
    red_r_list = []

    yellow_b_list = []
    yellow_g_list = []
    yellow_r_list = []

    for index_h in range(len(pos_h)) :
        for index_w in range(len(pos_w)) :

            sum_b = []
            sum_g = []
            sum_r = []
            for index_point in range(point_num) :
                
                if (index_h+index_w) % 2 == 0 : # red point
                    sum_b.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                    sum_g.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                    sum_r.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
                else : # yellow point
                    sum_b.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                    sum_g.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                    sum_r.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
                    
                #cv2.circle(copy_image, (pos_w[index_w],pos_h[index_h]-index_point) , 3 , (255,255,255), -1)
            
    
            if (index_h+index_w) % 2 == 0 : # red token
                print('red')
                red_b_list.append( sum(sum_b)/point_num )
                red_g_list.append( sum(sum_g)/point_num )
                red_r_list.append( sum(sum_r)/point_num )
            else : # yellow token
                print('yellow')
                yellow_b_list.append( sum(sum_b)/point_num )
                yellow_g_list.append( sum(sum_g)/point_num )
                yellow_r_list.append( sum(sum_r)/point_num )

    epsilon = 5

    red_b_low = min(red_b_list) - epsilon
    red_g_low = min(red_g_list) - epsilon
    red_r_low = min(red_r_list) - epsilon

    red_b_up = max(red_b_list) + epsilon
    red_g_up = max(red_g_list) + epsilon
    red_r_up = max(red_r_list) + epsilon

    yellow_b_low = min(yellow_b_list) - epsilon
    yellow_g_low = min(yellow_g_list) - epsilon
    yellow_r_low = min(yellow_r_list) - epsilon

    yellow_b_up = max(yellow_b_list) + epsilon
    yellow_g_up = max(yellow_g_list) + epsilon
    yellow_r_up = max(yellow_r_list) + epsilon

    red_mask_low = [red_b_low,red_g_low,red_r_low]
    red_mask_up = [red_b_up,red_g_up,red_r_up]

    yellow_mask_low = [yellow_b_low,yellow_g_low,yellow_r_low]
    yellow_mask_up = [yellow_b_up,yellow_g_up,yellow_r_up]

    return red_mask_low,red_mask_up,yellow_mask_low,yellow_mask_up
    
    
    

#red_mask_low,red_mask_up,yellow_mask_low,yellow_mask_up = calibration(path)



def image_processing(frame,board,red_mask_low,red_mask_up,yellow_mask_low,yellow_mask_up) :

    #board = np.zeros((6,7),dtype=np.int8)
    #print(board)

    raw_frame = cv2.flip(frame,1)
    hsv_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)

    pos_w = [60,150,250,340,430,520,610]
    pos_h = [55,150,230,310,390,460]

    point_num = 5

    for index_h in range(len(pos_h)) :
        for index_w in range(len(pos_w)) :
        
            sum_b = []
            sum_g = []
            sum_r = []
        
            for index_point in range(point_num) :
                sum_b.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][0] )
                sum_g.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][1] )
                sum_r.append( hsv_frame[pos_h[index_h]-index_point][pos_w[index_w]][2] )
            
            ch_b = sum(sum_b)/point_num
            ch_g = sum(sum_g)/point_num
            ch_r = sum(sum_r)/point_num
                          
            if (ch_b >= red_mask_low[0] and ch_b <= red_mask_up[0]) and (ch_g >= red_mask_low[1] and ch_g <= red_mask_up[1]) and (ch_r >= red_mask_low[2] and ch_r <= red_mask_up[2]) : board[index_h][index_w] = 2 # red detected = 2
            
            elif (ch_b >= yellow_mask_low[0] and ch_b <= yellow_mask_up[0]) and (ch_g >= yellow_mask_low[1] and ch_g <= yellow_mask_up[1]) and (ch_r >= yellow_mask_low[2] and ch_r <= yellow_mask_up[2]) : board[index_h][index_w] = 1 # yellow detected = 1

    print(board)
    
    return board
