import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import time
import torchvision.transforms as transforms
from model import CNN
from utils import num_graph, image_to_model, get_canvas_image, create_canvas, load_cnn_model, get_device

st.set_page_config(
    page_title="Digit Recognition Using CNN",
    page_icon="🔢",
    layout="wide",# 'centered'
    initial_sidebar_state="auto",
)


st.write("# Digit Recognition Using CNN's")
st.write('### Using a Convolutional Neural Network model from `PyTorch` trained on MNIST dataset')

# get device for the pytorch
device = get_device()

Network = load_cnn_model()

left_col, right_col = st.columns(2)

realtime_update = st.sidebar.checkbox("Realtime Update for the Canvas", value=True)
pen_thickness = st.sidebar.slider("Pen thickness: ", 1, 25, 9)

with left_col:
    st.write("#### Draw a single number between 0-9 in the box below.")
    # Creating the canvas
    canvas_result = create_canvas(pen_thickness=pen_thickness, realtime_update=realtime_update)






if canvas_result.image_data is not None:
    
    output_image = get_canvas_image(canvas_result=canvas_result)
    
    # Now we need to resize it, however doing so causes issues with the default arguments because it alters the range of pixel values to negative or positive.
    compressed_output_image = output_image.resize((22,22), Image.NEAREST) # Image.BILINEAR works too
    # transform image to tensor
    image_to_tensor = transforms.ToTensor()
    tensor_image = image_to_tensor(compressed_output_image)

    # padding
    tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
    # normalization
    normalize = transforms.Normalize(mean=0.1307, std=0.3081) # mean and standard deviation values for MNIST dataset
    tensor_image = normalize(tensor_image)
    # st.write(tensor_image.shape)

    
    
    
    # saving the tensor image 
    plt.imsave('processed_tensor.png',tensor_image.reshape(28,28), cmap='gray')

    
    model_image =Image.open('processed_tensor.png').resize((200,200), )
    # with left_col:
        
        
    #     st.write('Image from canvas')
    #     st.image(canvas_result.image_data)

    
    
    

    
    

    # prediction
    with torch.no_grad():
        certainty, number, output = image_to_model(tensor_image, Network)



        with right_col:
            if torch.sum(tensor_image) <= 0:
                st.write("#### Please draw a digit")
            else:

                with left_col:
                    st.write("#### What model sees")
                    st.image(model_image)

                st.write(f"#### Predicted as `{number}` with {certainty:.2f}% certainty.",)
                number_graph = num_graph(output)

                st.pyplot(number_graph)
        

