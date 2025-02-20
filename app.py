import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import model_loader  # Asegúrate de que esta importación es correcta

# Carga el modelo solo una vez en Streamlit
@st.cache_resource
def get_model():
    return model_loader.load_segmentation_model()

model = get_model()

# Configuración de la interfaz
st.title("Segmentación de Imágenes con DeepLabV3")
st.write("Sube una imagen y el modelo la segmentará.")

# Cargar imagen del usuario
uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Cargar imagen y mostrarla
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen original", use_container_width=True)

    # Transformar la imagen para el modelo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = transform(image).unsqueeze(0).to(device)  # Agregar batch dimension y mover a GPU si está disponible

    # Hacer predicción
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    
    # Convertir salida en máscara
    mask = output.argmax(0).byte().cpu().numpy()
    
    # Mostrar máscara sobre la imagen original
    st.image(mask, caption="Máscara de Segmentación", use_container_width=True)
