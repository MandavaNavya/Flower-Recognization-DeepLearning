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
from numpy import array
import cv2
from os import rename, listdir


# importing my data set
def load_data():
    img_array = []
    path1 = 'C:\\Flowers Data\\flowers'
    mypaths = []
    x = []
    class_num = []
#    d = {}
#    train_data = []
#    test_data = []
#    re_img = []
#    app_n = []
#    d1 = {}
    full_inp_data = []
    desired_size = 506
    dict_flower = {"daisy":0,"dandelion":1,"rose":2, "sunflower": 3, "tulip": 4}
    for f in os.listdir(path1): 
       dirs = os.listdir(path1)
       s = os.path.join(path1, f)
       mypaths.append(s)
       print(mypaths)
#       print(mypaths)
       class_num = dirs.index(f)
       print(class_num)
       
#       c = count()
#       numbers = [d.setdefault(i, next(c)) for i in mypaths]
##       print(numbers)
#       print(" i will execute my loop")
#       print(c)
       
       print(f)
       assing_val = dict_flower[f]
       print("assignment val: "+str(assing_val))
       for fp in os.listdir(s):
#           print("second loop")     
#           if os.path.isdir(s):
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
           
#           new_im = Image.new("RGB", (desired_size, desired_size))
#           new_im.paste(img_data, ((desired_size-new_size[0])//2,
#                    (desired_size-new_size[1])//2))
           """resizing the images by using imageOps module"""
           delta_w = desired_size - new_size[0]
           delta_h = desired_size - new_size[1]
           padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
           new_im = ImageOps.expand(img_data, padding)
#           print((new_im))
#           print(new_im)
           gs_data = new_im.convert(mode = 'L')
#           print(np.shape(gs_data))
           data = np.asarray(gs_data) 
           img_array = Image.fromarray(data) 
           
#           images.append(img_array)
#           print(np.shape(images))
#           images = np.asarray(img_array)
#           x.append(img_array)
           inpt = np.array(img_array)
#           print('Min: %.3f, Max: %.3f' % (inpt.min(), inpt.max()))
           pixels = asarray(inpt)
#           print('Data Type: %s' % pixels.dtype)
           """confirm image range in between 0 to 255"""
#           print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
           """convert intergers to float"""
           inpt = pixels.astype('float32')
           """ normalize to the range 0 to 1""" 
           inpt /= 255.0
           """ confirm the normalization"""
#           print('Min: %.3f, Max: %.3f' % (inpt.min(), inpt.max()))
#           print(inpt[100])
#           break
#           print(np.shape())
#           print(np.shape(inpt))
           inpt = np.array([np.reshape(inpt,np.prod(inpt.shape))])
#           print("inp: "+str(np.shape(inpt)))
           outp = vectorized_result(assing_val)
#           print("out: "+str(np.shape(outp)))
           x = [inpt,outp]
#           print(x)
#           print(np.shape(x))
           
           full_inp_data.append(x)
           
#          
    print("final training data" )
#    full_inp_data = np.array(full_inp_data)
    print(np.shape(full_inp_data))
    training_data, test_data = full_inp_data[:3000],full_inp_data[3000:]
    print(np.shape(training_data))
    print(np.shape(test_data))
#    print(np.shape(train_data))
    return training_data, test_data

#def load_data_wrapper():
#    tr_d, te_d= load_data()
##    print((tr_d[0]))
##    s = len(tr_d)
##    print(len(tr_d))S
#    training_data = [tr_d]
#    test_data = [te_d]

    
    return training_data, test_data
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((5, 1))
    e[j] = 1.0
    return e

#if __name__ == "__main__":
#    load_data()
           
           