import pandas as pd
import numpy as np
import cv2
import os


from os.path import isfile, join
from keras import backend as K
from os import listdir
from PIL import Image
from tensorflow.keras.models import model_from_json

def extract_imgs(img):
    img = ~img # Invert the bits of image 255 -> 0
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # Set bits > 127 to 1 and <= 127 to 0
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0]) # Sort by x

    img_data = []
    rects = []
    for c in cnt :
        x, y, w, h = cv2.boundingRect(c)
        rect = [x, y, w, h]
        rects.append(rect)

    bool_rect = []
    # Check when two rectangles collide
    for r in rects:
        l = []
        for rec in rects:
            flag = 0
            if rec != r:
                if r[0] < (rec[0] + rec[2] + 10) and rec[0] < (r[0] + r[2] + 10) and r[1] < (rec[1] + rec[3] + 10) and rec[1] < (r[1] + r[3] + 10):
                    flag = 1
                l.append(flag)
            else:
                l.append(0)
        bool_rect.append(l)

    dump_rect = []
    # Discard the small collide rectangle
    for i in range(0, len(cnt)):
        for j in range(0, len(cnt)):
            if bool_rect[i][j] == 1:
                area1 = rects[i][2] * rects[i][3]
                area2 = rects[j][2] * rects[j][3]
                if(area1 == min(area1,area2)):
                    dump_rect.append(rects[i])

    # Get the final rectangles
    final_rect = [i for i in rects if i not in dump_rect]
    for r in final_rect:
        x = r[0]
        y = r[1]
        w = r[2]
        h = r[3]

        im_crop = thresh[y:y+h+10, x:x+w+10] # Crop the image as most as possible
        im_resize = cv2.resize(im_crop, (28, 28)) # Resize to (28, 28)
        im_resize = np.reshape(im_resize, (1, 28, 28)) # Flat the matrix
        img_data.append(im_resize)

    return img_data

class ConvolutionalNeuralNetwork:
    def __init__(self):
        if os.path.exists('model/mod2.h5') and os.path.exists('model/mod2.json'):
            self.load_model()
            print("Inside const")

    def load_model(self):
        print('Loading Model...')
        model_json = open('model/mod2.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()
        loaded_model = model_from_json(loaded_model_json)

        print('Loading weights...')
        loaded_model.load_weights("model/mod2.h5")

        self.model = loaded_model


    def predict(self, operationBytes):
        print("Inside predict!")

        pil_image = Image.open(operationBytes).convert('L')
        open_cv_image = np.array(pil_image)
        print("Shape :" , open_cv_image.shape)

        open_cv_image = open_cv_image[:, :].copy() 

        print('Auxillary Image Saved!')

        # img = cv2.imread('aux.png',cv2.IMREAD_GRAYSCALE)
        img = open_cv_image

        print('Auxillary Image Loaded!')

        # os.remove('aux.png')
        if img is not None:
            img_data = extract_imgs(img)
            s=''
            for i in range(len(img_data)):
                img_data[i]=np.array(img_data[i])
                img_data[i]=img_data[i].reshape(-1,28,28,1)
                loaded_model = self.model
                result=loaded_model.predict_classes(img_data[i])
                if(result[0]==10):
                    s=s+'-'
                if(result[0]==11):
                    s=s+'+'
                if(result[0]==12):
                    s=s+'*'
                if(result[0]==0):
                    s=s+'0'
                if(result[0]==1):
                    s=s+'1'
                if(result[0]==2):
                    s=s+'2'
                if(result[0]==3):
                    s=s+'3'
                if(result[0]==4):
                    s=s+'4'
                if(result[0]==5):
                    s=s+'5'
                if(result[0]==6):
                    s=s+'6'
                if(result[0]==7):
                    s=s+'7'
                if(result[0]==8):
                    s=s+'8'
                if(result[0]==9):
                    s=s+'9'
            return s