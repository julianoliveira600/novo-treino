import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Triagem Histopatológica com IA",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------
# 2. DOWNLOAD E CARREGAMENTO DO MODELO VIA GOOGLE DRIVE
# -------------------------------------------------------------
GDRIVE_FILE_ID = "1AR-GAa8DAdIGEmmXMOLW93hnDNgzmm9p"
MODEL_LOCAL_PATH = "inception_multiscale_best.keras"

@st.cache_resource
def load_classification_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        with st.spinner("Baixando pesos do modelo via Google Drive (apenas na 1ª execução)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
            
    if not os.path.exists(MODEL_LOCAL_PATH):
        st.error("Falha ao obter o arquivo do modelo. Verifique se o link no Google Drive está com acesso público.")
        st.stop()
        
    # Mapeamento completo de possíveis objetos/funções usadas no treinamento
    custom_dict = {
        "preprocess_input": tf.keras.applications.inception_v3.preprocess_input,
        "Functional": keras.models.Functional,
        "Model": keras.Model
    }

    # Desativa checagem estrita de objetos seguros
    keras.config.enable_unsafe_deserialization()

    try:
        return keras.saving.load_model(
            MODEL_LOCAL_PATH,
            custom_objects=custom_dict,
            compile=False,
            safe_mode=False
        )
    except Exception:
        return keras.models.load_model(
            MODEL_LOCAL_PATH,
            custom_objects=custom_dict,
            compile=False,
            safe_mode=False
        )

model = load_classification_model()

# -------------------------------------------------------------
# 3. BARRA LATERAL (CONFIGURAÇÕES E MÉTRICAS)
# -------------------------------------------------------------
st.sidebar.title("🔬 Parâmetros do Sistema")
st.sidebar.markdown("**Arquitetura:** InceptionV3 Multiescala")
st.sidebar.markdown("**Entrada:** 299 × 299 px")

threshold = st.sidebar.slider(
    "Limiar de Decisão (Threshold):",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="O limiar de 0.40 foi calibrado para priorizar sensibilidade diagnóstica."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Métricas de Validação Independente")
st.sidebar.write("- **Acurácia Global:** 94,81%")
st.sidebar.write("- **ROC AUC:** 0,9893")
st.sidebar.write("- **Sensibilidade:** 91,38%")
st.sidebar.write("- **Especificidade:** 98,96%")

# -------------------------------------------------------------
# 4. ÁREA PRINCIPAL E FLUXO DE INFERÊNCIA
# -------------------------------------------------------------
st.title("Sistema de Auxílio ao Diagnóstico Histopatológico (H&E)")
st.markdown("Plataforma computacional para classificação e explicabilidade visual de lâminas teciduais.")

uploaded_file = st.file_uploader(
    "Envie a imagem histopatológica para triagem...", 
    type=["jpg", "png", "jpeg", "tif", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("1. Lâmina Enviada")
        st.image(image, use_container_width=True)

    # Pré-processamento
    img_resized = image.resize((299, 299))
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

    # Predição
    with st.spinner("Processando diagnóstico..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])
        is_malignant = prob >= threshold
        diagnostico = "Maligno" if is_malignant else "Benigno"

    with col2:
        st.subheader("2. Laudo Automatizado")
        if is_malignant:
            st.error(f"Classificação: **{diagnostico}**")
        else:
            st.success(f"Classificação: **{diagnostico}**")

        st.metric("Probabilidade de Malignidade", f"{prob * 100:.2f}%")
        st.progress(prob)
        st.caption(f"Critério clínico adotado: Maligno se P ≥ {threshold:.2f}")

    with col3:
        st.subheader("3. Explicabilidade (Grad-CAM)")
        try:
            last_conv = model.get_layer("mixed7")
            penultimate = model.get_layer("dense")
            classifier = model.get_layer("classifier")
            grad_model = keras.Model(inputs=model.inputs, outputs=[last_conv.output, penultimate.output])

            with tf.GradientTape() as tape:
                conv_out, dense_out = grad_model(img_array, training=False)
                tape.watch(conv_out)
                w, b = classifier.get_weights()
                logit = tf.matmul(dense_out, w) + b
                target = logit[0, 0] if logit[0, 0] >= 0 else -logit[0, 0]

            grads = tape.gradient(target, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.maximum(conv_out[0] @ pooled_grads[..., tf.newaxis], 0.0)
            heatmap = tf.squeeze(heatmap).numpy()
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            heatmap_smooth = cv2.GaussianBlur(heatmap, (3, 3), 0)
            heatmap_resized = cv2.resize(heatmap_smooth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.clip(heatmap_resized, 0, 1)

            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            superimposed = np.uint8(0.45 * heatmap_color + 0.55 * np.array(image))
            st.image(superimposed, use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM não pôde ser gerado para esta camada: {e}")