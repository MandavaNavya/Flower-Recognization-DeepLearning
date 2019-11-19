# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:43:19 2019

@author: raona
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 03:27:48 2019

@author: raona
"""

import PIL
from PIL import Image, ImageOps
from matplotlib import image 
from matplotlib import pyplot
from numpy import asarray
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import pandas as pd
import os, os.path
from itertools import count
from array import array
import cv2


# importing my data set
def load_data():
#    image_list = []
    path1 = 'C:\\Flowers Data\\flowers'
    mypaths = []
#    images = []
#    valid_images = [".jpg", ".gif", ".png", ".tga"]
    d = {}

#    d1 = {}
    desired_size = 506
    for f in os.listdir(path1): 
#        
#       for k, v in d.items():
#           if k in f:
#               print(v)
#       ext = os.path.splitext(f)[0]
###       print(ext)
#       if ext.lower() in valid_images:
##           
#           continue 
#       print(ext.lower(), "i got my extention")
       
       s = (os.path.join(path1, f))
       mypaths.append(s)
       print(mypaths)
#       print(mypaths)
       """ assigning an number for every unique path"""
#       d = {num : init for init, num in enumerate(set(mypaths))}
#       numbers = [d[num] for num in mypaths ]
#       print(numbers)
       
#       d = {}
       c = count()
       numbers = [d.setdefault(i, next(c)) for i in mypaths]
#       print(numbers)
#       print(c)
       #       print(s, "got my every path")
       for fp in os.listdir(s):
#          
           try:
#            print(fp, 'i got fp')
             img_data = Image.open((os.path.join(s, fp)))
           except IOError:
             pass
         
           old_size = img_data.size
           ratio = float(desired_size)/max(old_size)
#           print(ratio)
           new_size = tuple([int(x*ratio) for x in old_size])
           """ resizing the image by using image module"""
           img_data = img_data.resize(new_size, Image.ANTIALIAS)
           
           new_im = Image.new("RGB", (desired_size, desired_size))
           new_im.paste(img_data, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
           """resizing the images by using imageOps module"""
           delta_w = desired_size - new_size[0]
           delta_h = desired_size - new_size[1]
           padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
           new_im = ImageOps.expand(img_data, padding)
           
           
#           new_im.show()
       training_data, test_data = train_test_split(new_im, test_size = 0.25)
#       training_data, test_data  = train_test_split(images, test_size = 0.25)
#       print("printing training data:")
       print(np.shape(training_data))
#       tra_data = asarray(training_data)
#       tra_array = Image.fromarray(tra_data)
    print("final training data")
#    print(len(training_data), "my training data is")
#    print(len(test_data))
    return(training_data, test_data , numbers)
    
    
def load_data_wrapper():
#    training_inputs = []
#    test_inputs = []
    tr_d, te_d, numbers = load_data()
#    print(len(tr_d))
#    shape = [500, 500, 500]
#    i = 0
#    for i in range(len(tr_d)):
#       constant = cv2.copyMakeBorder(tr_d[i],30,30,40,40,cv2.BORDER_CONSTANT, value=shape)
#       print(np.shape(constant))
        
        
#    print(len(te_d))
#    print(te_d[0])
#    print(tr_d[1])
#    print(tr_d[0])
#    print(np.shape(tr_d))
#    print(np.shape(te_d))
#    print(np.shape(tr_d))
#    print(np.shape(tr_d))
    tr_array = asarray(tr_d)
    tr_nparr = Image.fromarray(tr_array)
#    print(tr_nparr.mode)
#    tr_d
    train_inputs = np.pad(tr_nparr, ((5, 5), (2, 4)),'constant', constant_values =(0, 0))
    print(np.shape(train_inputs))
    train_result = [vectorized_result(y) for y in numbers]
    print(type(train_result), "got my result")
    training_data = list(zip(train_inputs, train_result))
    print(np.shape(training_data))
#    print(train_data, "i got train data finally:")
#    print(np.shape(train_data))
    te_array = asarray(te_d)
    te_nparr = Image.fromarray(te_array)
#    print(te_nparr.mode)
#    
    test_inputs = [np.pad(te_nparr, ((5, 5), (2, 4)), "constant", constant_values = (0, 0))]
    test_data = list(zip(test_inputs, te_d[1]))
    print(np.shape(test_data))

#    print(test_data, "i got my test data")
#    print(np.shape(test_data))
    return(training_data, test_data)
#    
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((5,1))
#    print(j)
#    print(e)
    e[j] = 1.0
    return e
    