import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import os
import gdown

st.set_page_config(
    page_title="Triagem Histopatológica com IA",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------
# CONFIGURAÇÃO DO MODELO NO GOOGLE DRIVE
# -------------------------------------------------------------
GDRIVE_FILE_ID = "1AR-GAa8DAdIGEmmXMOLW93hnDNgzmm9p"
MODEL_LOCAL_PATH = "inception_multiscale_best.keras"

@st.cache_resource
def load_classification_model():
    # Faz o download automático caso o arquivo não exista no container
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("Baixando pesos do modelo via Google Drive (apenas na 1ª inicialização)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
            
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.error("Falha ao baixar o modelo. Verifique se o compartilhamento no Drive está marcado como 'Qualquer pessoa com o link'.")
        st.stop()
        
    return keras.models.load_model(MODEL_LOCAL_PATH)

# Carrega e mantém em cache
model = load_classification_model()