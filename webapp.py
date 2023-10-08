import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import requests
from model import MainModel


def load_image(image):
    image = Image.open(image).convert("L")  # Convert to grayscale
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image


@st.cache_data
def load_model():
    model = MainModel()
    model_path = Path("custom_GAN.pt")
    if not model_path.exists():
        with st.spinner("Downloading model... this may take awhile! ⏳"):
            url = 'https://storage.googleapis.com/new-york-cab-data/BnW%20Model/custom_GAN.pt'
            r = requests.get(url)
            model_path.write_bytes(r.content)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


model = load_model()
st.title("Image Colorization App")

uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    height, width = input_image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the model's expected input size
        transforms.ToTensor(),
    ])
    input_image = transform(input_image).unsqueeze(0)  # Add batch dimension
    colorized_image = model.predict(input_image)
    colorized_image = Image.fromarray((colorized_image * 255).astype(np.uint8)).resize((height, width // 2))
    st.image(colorized_image, caption="Colorized image", use_column_width=True)
