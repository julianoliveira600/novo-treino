import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import cv2
from PIL import Image
import os
import gdown
import zipfile
import json
import shutil

#-------------------------------------------------------------
# 1. Importação pdf
#-------------------------------------------------------------
from fpdf import FPDF
import io
import tempfile
from datetime import datetime

def generate_pdf_report(orig_img_pil, cam_img_np, filename, prob, threshold, diagnostico):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cabeçalho
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Relatório de Triagem Histopatológica", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align="C")
    pdf.ln(8)
    
    # Metadados e Parâmetros
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Dados do Exame", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Arquivo da Lâmina: {filename}", ln=True)
    pdf.cell(0, 6, "Arquitetura do Modelo: InceptionV3 Multiescala", ln=True)
    pdf.cell(0, 6, f"Limiar Diagnóstico Adotado: {threshold:.2f}", ln=True)
    pdf.ln(5)
    
    # Diagnóstico Automatizado
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Resultado Automatizado", ln=True)
    pdf.set_font("Helvetica", "B", 11)
    
    # Cor do texto: Vermelho para Maligno, Verde para Benigno
    if diagnostico == "Maligno":
        pdf.set_text_color(180, 0, 0)
    else:
        pdf.set_text_color(0, 130, 0)
    pdf.cell(0, 7, f"Classificação Estimada: {diagnostico}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Probabilidade de Malignidade: {prob * 100:.2f}%", ln=True)
    pdf.ln(8)
    
    # Imagens do Exame lado a lado
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Registro Visual e Explicabilidade (Grad-CAM)", ln=True)
    pdf.ln(2)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_orig, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_cam:
        
        orig_img_pil.save(f_orig.name, format="PNG")
        Image.fromarray(cam_img_np).save(f_cam.name, format="PNG")
        
        y_pos = pdf.get_y()
        pdf.image(f_orig.name, x=15, y=y_pos, w=85)
        pdf.image(f_cam.name, x=110, y=y_pos, w=85)
        
        # Legendas
        pdf.set_y(y_pos + 90)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(85, 6, "Lâmina Original (H&E)", align="C")
        pdf.cell(10, 6, "")
        pdf.cell(85, 6, "Mapa de Atenção Grad-CAM", align="C")
        
        os.remove(f_orig.name)
        os.remove(f_cam.name)
        
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Nota: Este laudo foi gerado por um modelo de aprendizado profundo para triagem computacional de apoio e não substitui a avaliação microscópica definitiva realizada por um médico patologista.")
    
    return bytes(pdf.output())

# -------------------------------------------------------------
# 1.1 CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Triagem Histopatológica com IA",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------
# 2. DOWNLOAD E CORREÇÃO AUTOMÁTICA DO MODELO
# -------------------------------------------------------------
GDRIVE_FILE_ID = "1AR-GAa8DAdIGEmmXMOLW93hnDNgzmm9p"
MODEL_RAW_PATH = "model_raw.keras"
MODEL_PATCHED_PATH = "inception_multiscale_fixed.keras"

def strip_quantization_keys(obj):
    """Remove recursivamente qualquer chave quantization_config do JSON do modelo."""
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for k, v in list(obj.items()):
            strip_quantization_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_quantization_keys(item)

def patch_keras_zip(src_zip, dst_zip):
    """Abre o arquivo .keras, limpa o config.json e regrava pronto para uso."""
    tmp_dir = "/tmp/keras_patch"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(src_zip, 'r') as zin:
        zin.extractall(tmp_dir)

    cfg_path = os.path.join(tmp_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        strip_quantization_keys(cfg)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    with zipfile.ZipFile(dst_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for root, _, files in os.walk(tmp_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, tmp_dir)
                zout.write(full_p, rel_p)

    shutil.rmtree(tmp_dir)

@st.cache_resource
def load_classification_model():
    if not os.path.exists(MODEL_PATCHED_PATH):
        if not os.path.exists(MODEL_RAW_PATH):
            with st.spinner("Baixando pesos do modelo via Google Drive (apenas na 1ª inicialização)..."):
                url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                gdown.download(url, MODEL_RAW_PATH, quiet=False)
                
        with st.spinner("Adaptando compatibilidade de serialização..."):
            patch_keras_zip(MODEL_RAW_PATH, MODEL_PATCHED_PATH)

    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    custom_objects = {
        "preprocess_input": tf.keras.applications.inception_v3.preprocess_input
    }

    return keras.models.load_model(
        MODEL_PATCHED_PATH,
        custom_objects=custom_objects,
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

    st.markdown("---")
    pdf_data = generate_pdf_report(
        orig_img_pil=image,
        cam_img_np=superimposed,
        filename=uploaded_file.name,
        prob=prob,
        threshold=threshold,
        diagnostico=diagnostico
    )

    st.download_button(
        label="📄 Baixar Laudo Diagnóstico em PDF",
        data=pdf_data,
        file_name=f"laudo_{uploaded_file.name.split('.')[0]}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    with col1:
        st.subheader("1. Lâmina Enviada")
        st.image(image, use_container_width=True)

    img_resized = image.resize((299, 299))
    img_array = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)

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
            
            # Ponderação dos canais convolucionais
            heatmap = tf.maximum(conv_out[0] @ pooled_grads[..., tf.newaxis], 0.0)
            heatmap = tf.squeeze(heatmap).numpy()

            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # --- MELHORIA DE CONTRASTE E ISOLAMENTO DE FOCOS ---
            # Remove ativações fracas (abaixo do limiar de relevância)
            threshold_activation = 0.35
            heatmap_filtered = np.where(heatmap >= threshold_activation, heatmap, 0.0)

            # Redimensionamento suave para o tamanho nativo da imagem
            heatmap_smooth = cv2.GaussianBlur(heatmap_filtered, (5, 5), 0)
            heatmap_resized = cv2.resize(heatmap_smooth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.clip(heatmap_resized, 0, 1)

            # Colormap JET (Azul -> Verde -> Amarelo -> Vermelho)
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Máscara de transparência progressiva: onde a ativação é zero, não pinta nada
            alpha_mask = np.expand_dims(heatmap_resized, axis=-1)
            orig_img_np = np.array(image, dtype=np.float32)
            heatmap_color_float = heatmap_color.astype(np.float32)

            # Sobreposição ponderada com foco nítido
            blend = orig_img_np * (1.0 - 0.65 * alpha_mask) + heatmap_color_float * (0.65 * alpha_mask)
            superimposed = np.clip(blend, 0, 255).astype(np.uint8)

            st.image(superimposed, use_container_width=True)
            st.caption("Focos quentes (amarelo/vermelho) indicam áreas determinantes para a classificação.")
        except Exception as e:
            st.warning(f"Grad-CAM não pôde ser gerado para esta camada: {e}")