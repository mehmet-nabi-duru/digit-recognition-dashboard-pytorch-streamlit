import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import time
import torchvision.transforms as transforms
import cv2
import seaborn as sns
from streamlit_drawable_canvas import st_canvas


############### MODEL ###############
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*8*8, 600)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(600, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu1(x)
        x = self.maxpool2(x)

        x = x.view(-1, 32*8*8)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
############# FUNCTIONS #############
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

st.set_page_config(
    page_title="Digit Recognition Using CNN",
    page_icon=":1234:",
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
        

