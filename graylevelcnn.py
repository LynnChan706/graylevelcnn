#!/usr/bin/env python2.7
# coding=utf-8

'''
@date = '17/4/7'
@author = 'chenliang'
@email = ''
'''

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json

import Image, ImageFont, ImageDraw
import cv2
import random
import numpy as np
import json
from math import *
from json import load


batch_size = 128
num_classes = 10
epochs = 10

img_rows = 128
img_cols = 20


def dataGenerate(num):
    img = Image.new("RGB", (img_rows, img_cols), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    line_w = img_rows/num

    for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        s_l=line_w*i+line_w/2
        e_l=line_w*(i+1)+1+line_w/2
        if i < num-1:
            draw.line([s_l,0,s_l,img_cols],(r,g,b),line_w+1)
        else:
            draw.line([s_l+2, 0, s_l+2, img_cols], (r, g, b), img_rows+2-i*line_w)
    img1 = np.array(img)
    return img1


def rot(img,angel,shape,max_angel):
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*cos((float(max_angel)/180) * 3.14)),shape[0])

    interval = abs(int(sin((float(angel)/180) * 3.14)* shape[0]))

    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]],[size[0],0],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]],[size[0]-interval,0],[size[0],size_o[1]]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)
    return dst


def rotRandrom(img, factor, size):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size)
    return dst


def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2)
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7)
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8)

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img


def AddNoiseSingleChannel(single):
    diff = 30
    noise = np.random.normal(0,1,single.shape)
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise = diff*noise
    noise = noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] = AddNoiseSingleChannel(img[:,:,0])
    img[:,:,1] = AddNoiseSingleChannel(img[:,:,1])
    img[:,:,2] = AddNoiseSingleChannel(img[:,:,2])
    return img

def r(val):
    return int(np.random.random() * val)

def getImageandLabel(num):
    imglist=[]
    labellist=[]
    for j in range(num):
        i=random.randint(4,9)
        img=dataGenerate(i)
        img = rot(img, r(12) - 6, img.shape, 3)
        img = rotRandrom(img, 1, (img.shape[1], img.shape[0]))
        img = tfactor(img)
        img = addNoise(img)
        img = cv2.GaussianBlur(img, (r(5) / 2 * 2 + 1, r(5) / 2 * 2 + 1), 0)
        img = cv2.resize(img, (img_rows, img_cols))

        imglist.append(img)
        labellist.append(i)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    imgarray=np.array(imglist)
    labelarray = np.array(labellist)

    return imgarray,labelarray

def trainData(num=10000):
    input_shape = (img_cols,img_rows,3)

    x_train, y_train = getImageandLabel(num)
    x_test, y_test = getImageandLabel(num/10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print("Now we save model")
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def testdata():
    instr=''
    with open("model.json", "r") as infile:
        instr=load(infile)
    model=model_from_json(instr)
    model.load_weights("model.h5")
    x,y = getImageandLabel(1)
    print y
    res = model.predict(x)
    print res

if __name__ == '__main__':
    trainData()
    # testdata()
    # x, y = getImageandLabel(1)
    # print y

    # input_shape1 = (img_cols, img_rows, 3)
    # model = Sequential()
    # model.input_layers()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape1))   #卷积  128*20 126*18
    # model.add(Conv2D(64, (3, 3), activation='relu')) #卷积  124*16
    # model.add(MaxPooling2D(pool_size=(2, 2))) # 62*8
    # model.add(Dropout(0.25))
    # model.add(Flatten()) #62*8=496
    # outshape=model.compute_output_shape(input_shape=input_shape1)
    # print outshape







