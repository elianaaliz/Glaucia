import streamlit as st

from PIL import Image
import base64
import torch
from torchvision.transforms import transforms
import torchvision.models as models
import time

def get_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    .title-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .title-wrapper h1 {
        font-size: 122px; /* Ajusta el tamaño de la fuente según tus necesidades */
        color: white; /* Cambia el color del texto según tus necesidades */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Añade sombra al texto */
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Llamar a la función con la ruta de la imagen
set_background("images/back2.png")


# Mostrar el logo y el nombre de la aplicación en dos columnas
col1, col2 = st.columns([1, 2])
with col1:
    st.image("images/logo2.png", width=380)  # Ajusta el valor de width según tus necesidades
with col2:
    st.markdown('<div class="title-wrapper"><h1> Glauc<span style="color: #48B4B8;">ia</span></h1></div>', unsafe_allow_html=True)


# Definir el mapa de etiquetas
label_map = {
    0: "Glaucoma Positiva",
    1: "Glaucoma Negativa",
}

# Preprocesamiento de la imagen
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Añadir dimensión de lote
    return input_batch

# Clasificación de la imagen
def classify_image(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        return predicted.item(), output

def main():
    st.title("Detección de Glaucoma")

    # Cargar el modelo EfficientNet-V2
    class_size = 2
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = torch.nn.Linear(1280, class_size)
    model.load_state_dict(torch.load("models/best.pth", map_location=torch.device('cpu')))  # Cargar el modelo pre-entrenado en la CPU
    model.eval()

    # Subida de la imagen
    uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png"])

    # Banner para seleccionar imágenes de muestra
    st.markdown("---")
    st.subheader("Imágenes de muestra:")
    row = st.columns(4)
    sample_images = [
        "images/imagen_de_muestra_1.png",
        "images/imagen_de_muestra_2.png",
        "images/imagen_de_muestra_3.png",
        "images/imagen_de_muestra_4.png"
    ]
    for i, sample_image_path in enumerate(sample_images):
        image = Image.open(sample_image_path)
        if row[i].button(f"Imagen de muestra {i + 1}", key=f"imagen_muestra_{i + 1}"):
            uploaded_file = sample_image_path

    if uploaded_file is not None:
        left_column, right_column = st.columns([2, 3])
        
        # Mostrar la imagen subida y su predicción a la izquierda y derecha respectivamente
        with left_column:
            st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
        with right_column:
            input_batch = preprocess_image(uploaded_file)
            prediction, output = classify_image(model, input_batch)
            predicted_class = label_map[prediction]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item() * 100
            if predicted_class == "Glaucoma Positiva":
                st.write(f"<p style='font-size:20px; text-align:center; color:red;'>Predicción: {predicted_class}</p>", unsafe_allow_html=True)
            else:
                st.write(f"<p style='font-size:20px; text-align:center;'>Predicción: {predicted_class}</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size:20px; text-align:center;'>Confianza: {confidence:.2f}%</p>", unsafe_allow_html=True)
        
        # Tiempo de Inferencia
        start_time = time.time()
        prediction, _ = classify_image(model, input_batch)
        inference_time = time.time() - start_time
        st.write(f"<p style='font-size:20px; text-align:center;'>Tiempo de Inferencia: {inference_time:.4f} segundos</p>", unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("Por favor, sube una imagen o selecciona una de las imágenes de muestra.")

if __name__ == "__main__":
    main()
