import streamlit as st
import os
import numpy as  np
from PIL import Image
import pickle
import requests
import tensorflow
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list=np.array(pickle.load(open('featurevector.pkl','rb')))
filenames=pickle.load(open('filename.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion recommendation system')



def save_upload_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expand_img=np.expand_dims(img_array,axis=0)
    pre_img=preprocess_input(expand_img)
    result=model.predict(pre_img).flatten() 
    normalized=result/norm(result)
    return normalized



def recommend(features,feature_list):
    neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)
    distance,indices=neighbors.kneighbors([features])
    return indices



#file upload and save
uploaded_file=st.file_uploader('choose an fashion image')
if uploaded_file is not None:
    if save_upload_file(uploaded_file):
        #display the file
        display_image=Image.open(uploaded_file)
        st.image(display_image)
        #feature extract
        features=feature_extraction(os.path.join('uploaded',uploaded_file.name),model)
        #recommendaitons
        indices=recommend(features,feature_list)
        #show
        col1,col2,col3,col4,col5=st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header('some error occured in file upload')