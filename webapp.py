import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.color import lab2rgb
from pathlib import Path
import requests
from model import MainModel


# Function to load and preprocess the input image
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


def generate_colorized_image(model, input_image):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input({'L': input_image, 'ab': torch.zeros_like(input_image)})
        model.forward()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    return np.concatenate([real_imgs[0], fake_imgs[0]], axis=1)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1)
    L2 = Lab.permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb_imgs = []
    for img in L2:
        if img.shape[2] == 2:
            img = np.concatenate([img, np.zeros((256, 256, 1))], axis=2)
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


# Load your trained model
model = load_model()
# Streamlit App
st.title("Image Colorization App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    height, width = input_image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the model's expected input size
        transforms.ToTensor(),
    ])
    input_image = transform(input_image).unsqueeze(0)  # Add batch dimension
    colorized_image = generate_colorized_image(model, input_image)
    colorized_image = Image.fromarray((colorized_image * 255).astype(np.uint8)).resize((height, width//2))
    st.image(colorized_image, caption="Colorized image", use_column_width=True)
