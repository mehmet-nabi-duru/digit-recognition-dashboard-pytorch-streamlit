import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import CNN
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import cv2
import seaborn as sns

@st.cache
def get_device():
    # gets available device if cuda is enabled
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

@st.cache
def load_cnn_model():
    model = CNN()

    model.load_state_dict(torch.load("./models/model_weights.pth", map_location=torch.device("cpu")))
    return model

def create_canvas(height=200, width=200, pen_thickness=9, realtime_update=True):
    return st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=pen_thickness,
        stroke_color='#FFFFFF',
        background_color='#000000',
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode='freedraw',
        key="canvas",
    )



def get_canvas_image(canvas_result):
    try:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')

    except Exception as e:
        st.error(e, icon="🚨")
    
    try:
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2GRAY)
        height, width = image.shape
        x, y, w, h = cv2.boundingRect(image)
        ROI = image[y:y+h, x:x+w]
        mask = np.zeros([ROI.shape[0]+10, ROI.shape[1]+10])
        width, height = mask.shape
        x = width//2 - ROI.shape[0]//2
        y = height//2 - ROI.shape[1]//2
        mask[y:y+h, x:x+w] = ROI
        output_image = Image.fromarray(mask)

    except Exception as e:
        st.error(e, icon="🚨")


    return output_image

def image_to_model(image,model):
    image = image.reshape((1,1,28,28))
    output = model(image)

    softmax = nn.Softmax(dim=1)
    output = softmax(output)
    certainty, number = torch.max(output[0], 0)
    certainty = certainty.item()*100
    number = number.item()

    return certainty, number, output*100

def num_graph(output):
    outputdict = dict(enumerate(output[0].detach().numpy()))

    # fig = px.bar(x=list(outputdict.keys()), y=list(outputdict.values()),)

    fig = plt.figure(figsize=(7,4))
    sns.barplot(x=list(outputdict.keys()), y=list(outputdict.values()), )
    plt.xlabel("Numbers")
    plt.ylabel("Certainty")

    return fig