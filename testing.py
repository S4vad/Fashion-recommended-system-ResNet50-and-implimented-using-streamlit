import pickle
import tensorflow 
import numpy as np
from tensorflow.keras.layers import GloablMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list=np.array(pickle.load(open()))
filenames=pickle.load(open())

model=ResNet50(weights='imagenet',inlcude_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


img=cv2.imread('shirt.jpg')
img=cv2.resize(img,(224,224))
img=np.array(img)
expand_img=np.expand_dims(img,axis=0)
pre_img=preprocess_input(expand_img)
result=model.predict(pre_img).flatten() 
normalized=result/norm(result)

neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distance,indices=neighbors.Kneighbors([normalized])
print(indices)

for file in indices[0][:6]:
    temp_img=cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitkey(0)