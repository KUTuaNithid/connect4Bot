import cv2
import time
import numpy as np

class ImageProcessing:
    def __init__(self):
        self.image = np.zeros((6,7,3),dtype=np.int8)
        self.red_mask_low = []
        self.red_mask_up = []
        self.yellow_mask_low = []
        self.yellow_mask_up = []
    
    def capture(self):
        cap = cv2.VideoCapture(0)
        time.sleep(4)
        ret,frame = cap.read()
        time.sleep(2)
        cap.release()
        cv2.imwrite('capture.jpg',frame)
        self.image = frame
    
    def get_box(self,image,index_x,index_y):
        X = [0,67,166,261,360,457,554,639]
        Y = [0,75,159,243,328,411,480] 

        box_image = image[Y[index_y]:Y[index_y+1],X[index_x]:X[index_x+1],:]
        return box_image.copy()

    def pre_process(self):
        img = self.image
        img_copy = img.copy()

        input_pts = np.float32([[0,21],[35,463],[629,20],[590,465]])
        output_pts = np.float32([[0,0],[0,479],[639,0],[639,479]])
        
        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
 
        # Apply the perspective transformation to the image
        out = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
        cv2.imwrite('capture_per.jpg',out)
        # Display the transformed image
        blur = cv2.blur(out,(12,12))
        cv2.imwrite('capture_blur.jpg',blur)
        # img_hsv = cv2.cvtColor(img_copy,cv2.COLOR_BGR2HSV)
        # lower_red = np.array([110,50,50])
        # upper_red = np.array([130,255,255])
        # cv2.imshow('hsv_img',img_hsv)
        # cv2.imshow
        self.image = blur

    def calibration(self) :
        self.capture()
        self.pre_process()
        frame = self.image.copy()
        copy_image = frame.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #cv2.imshow('main image',self.image)
        

        for i in range(6):
            for j in range(2):
                box = self.get_box(hsv_frame,j*6,i)
                # copy_box = box.copy()
                box_h,box_w,_ = box.shape
                mid_h = round(box_h/2)-1
                mid_w = round(box_w/2)-1
                h_sample = np.random.randint(mid_h-6,mid_h+7,5)
                w_sample = np.random.randint(mid_w-6,mid_w+7,5)
                for h_pos in h_sample:
                    for w_pos in w_sample:
                        h_channel = box[h_pos,w_pos,0]
                        s_channel = box[h_pos,w_pos,1]
                        v_channel = box[h_pos,w_pos,2] 
                        # copy_box[h_pos,w_pos] = np.array([40,255,255])
                        
                        if (i+j) %2 == 0: # RED
                            # print(box[h_pos,w_pos],'RED')
                            inRange = cv2.inRange(box[h_pos,w_pos].reshape((1,1,3)),self.red_mask_low,self.red_mask_up)
                            if not inRange :
                                ref_mask_low = self.red_mask_low
                                ref_mask_up = self.red_mask_up
                        else:
                            # print(box[h_pos,w_pos],'YEL')
                            inRange = cv2.inRange(box[h_pos,w_pos].reshape((1,1,3)),self.yellow_mask_low,self.yellow_mask_up)
                            if not inRange :
                                ref_mask_low = self.yellow_mask_low
                                ref_mask_up = self.yellow_mask_up
                        
                        if not inRange :
                            if h_channel > ref_mask_up[0]:
                                ref_mask_up[0] = h_channel
                            if h_channel < ref_mask_low[0]:
                                ref_mask_low[0] = h_channel
                            if s_channel > ref_mask_up[1]:
                                ref_mask_up[1] = s_channel
                            if s_channel < ref_mask_low[1]:
                                ref_mask_low[1] = s_channel
                            if v_channel > ref_mask_up[2]:
                                ref_mask_up[2] = v_channel
                            if v_channel < ref_mask_low[2]:
                                ref_mask_low[2] = v_channel
                        
        # cv2.imshow('copy_box',copy_box)
        print('low red',self.red_mask_low)
        print('up red',self.red_mask_up)
        print('low yellow',self.yellow_mask_low)
        print('up yellow',self.yellow_mask_up)
        # cv2.waitKey(0)

    
    def process_image(self):
        self.capture()
        self.pre_process()
        frame = self.image.copy()
        board = np.zeros((6,7),dtype=np.int8)
        copy_image = frame.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imwrite('capture_hsv.jpg',hsv_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))

        for idx_x in range(7):
            for idx_y in range(6):
                box = self.get_box(hsv_frame,idx_x,idx_y)
                
                yellow_mask = cv2.inRange(box, self.yellow_mask_low, self.yellow_mask_up)
                yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
                
                red_mask = cv2.inRange(box, self.red_mask_low, self.red_mask_up)
                red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
                if np.sum(yellow_mask!=0) > 1000 :
                    board[idx_y,idx_x] = 1
                elif np.sum(red_mask!=0) > 1000:
                    board[idx_y,idx_x] = 2
                else:
                    board[idx_y,idx_x] = 0
        print(board)
        return board