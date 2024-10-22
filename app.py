import streamlit as st
import gdown
import requests
import os
import torchvision 
import torch
from PIL import Image
import numpy as np

file_id = '1G57svDNd16pFgHQvYpkmBZF3p5a2w6kR'
model_url = f'https://drive.google.com/uc?id={file_id}'
model_path = 'resnet_model.pth'

# Function to download file from Google Drive
def download_model(url, save_path):
    st.write("Attempting to download the model from Google Drive...")
    try:
        gdown.download(url, save_path, quiet=False)
        st.write("Model downloaded successfully.")
    except Exception as e:
        st.write(f"Failed to download the model. Error: {e}")

# Check if the model exists locally
if not os.path.exists(model_path):
    st.write("Model not found.")
    download_model(model_url, model_path)
else:
    st.write("Model found locally.")

# Load the model architecture
resnet_model = torchvision.models.resnet18(weights=None)  # Load the model structure without pre-trained weights
resnet_model.fc = torch.nn.Linear(512, 37)  # Change the output layer to have 37 classes
resnet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
resnet_model.eval()  # Set the model to evaluation mode

# Load class names from the file
with open('class_names.txt', 'r') as file:  # Update with your local path
    class_names = [line.strip() for line in file.readlines()]

# Define the transformation for the uploaded image
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # Resize to the input size of the model
    torchvision.transforms.ToTensor(),  # Convert to tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

st.title("Malaysian Sign Language Recognition")
st.write("This is a simple recognition web to predict Malaysian Sign Language.")
st.write("Please upload an image file for the model to make prediction.")
st.write("The predicted sign language label, confidence percentage of the model prediction as well as for each class will be displayed below.")
st.write("The model is trained on the classes A to Z and 0 to 10.")
st.write("And it will be predicted as corresponding labels of 0 to 36.")
st.write("Below shows the image of the dataset's data categories and theirs corresponding labels trained inside the model.")
st.image('Data Categories.png', caption='Data Categories', use_column_width=True) 

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)

    # Convert the image to RGB if it is grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = resnet_model(image_tensor)  # No need for .cuda() if on CPU
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted_class = np.argmax(probabilities.numpy())  # Get the predicted class index
        confidence = probabilities.max().item() * 100  # Get confidence percentage

    # Display the predicted class
    st.write(f"Predicted Label: {predicted_class}")
    st.write(f"Confidence Percentage: {confidence:.2f}%")
    st.write("Confidence Percentages for each class:")
    for i, probability in enumerate(probabilities[0]):
        st.write(f"{class_names[i]}: {probability.item() * 100:.2f}%")
