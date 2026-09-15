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
import tempfile
from datetime import datetime
from fpdf import FPDF

# -------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# -------------------------------------------------------------
st.set_page_config(
    page_title="Triagem Histopatológica com IA",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------------------------------------
# 2. FUNÇÃO DE GERAÇÃO DE RELATÓRIO PDF
# -------------------------------------------------------------
def generate_pdf_report(orig_img_pil, cam_img_np, filename, prob, threshold, diagnostico, info_paciente):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cabeçalho Principal
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Relatorio de Triagem Histopatologica Veterinaria", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 5, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    
    # 1. Dados do Paciente e Tutor
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. Identificacao do Paciente e Amostra", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    col_w1, col_w2 = 95, 95
    pdf.cell(col_w1, 6, f"Paciente: {info_paciente['nome'] or 'Nao informado'}", ln=False)
    pdf.cell(col_w2, 6, f"Tutor: {info_paciente['tutor'] or 'Nao informado'}", ln=True)
    
    pdf.cell(col_w1, 6, f"Especie: {info_paciente['especie']}", ln=False)
    pdf.cell(col_w2, 6, f"Raca: {info_paciente['raca'] or 'SRD / Nao informada'}", ln=True)
    
    pdf.cell(col_w1, 6, f"Local da Coleta: {info_paciente['topografia'] or 'Nao informado'}", ln=False)
    pdf.cell(col_w2, 6, f"Arquivo da Lamina: {filename}", ln=True)
    
    if info_paciente['obs']:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Observacoes Clinicas: {info_paciente['obs']}")
    pdf.ln(4)
    
    # 2. Diagnóstico Automatizado
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Parecer Computacional", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Arquitetura: InceptionV3 Multiescala (H&E 299x299px) | Limiar: {threshold:.2f}", ln=True)
    
    pdf.set_font("Helvetica", "B", 11)
    if diagnostico == "Maligno":
        pdf.set_text_color(180, 0, 0)
    else:
        pdf.set_text_color(0, 130, 0)
    pdf.cell(0, 6, f"Classificacao Sugerida: {diagnostico}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Indice de Confianca (Probabilidade de Malignidade): {prob * 100:.2f}%", ln=True)
    pdf.ln(4)
    
    # 3. Imagens do Exame
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Registro Visual e Explicabilidade (Grad-CAM)", ln=True)
    pdf.ln(1)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_orig:
        orig_img_pil.save(f_orig.name, format="PNG")
        y_pos = pdf.get_y()
        pdf.image(f_orig.name, x=15, y=y_pos, w=85)
        
        if cam_img_np is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_cam:
                Image.fromarray(cam_img_np).save(f_cam.name, format="PNG")
                pdf.image(f_cam.name, x=110, y=y_pos, w=85)
                os.remove(f_cam.name)
        
        pdf.set_y(y_pos + 90)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(85, 6, "Lamina Original (H&E)", align="C")
        if cam_img_np is not None:
            pdf.cell(10, 6, "")
            pdf.cell(85, 6, "Mapa de Atencao Grad-CAM (mixed7)", align="C")
            
        os.remove(f_orig.name)
        
    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Aviso Legal: Documento gerado por modelo computacional de triagem assistida (Deep Learning). Os resultados fornecidos possuem finalidade de suporte a decisao e devem ser correlacionados com dados clinicos, historico do animal e revisao macro/microscopica por medico veterinario patologista.")
    
    return bytes(pdf.output())

# -------------------------------------------------------------
# 3. DOWNLOAD E CORREÇÃO AUTOMÁTICA DO MODELO
# -------------------------------------------------------------
GDRIVE_FILE_ID = "1AR-GAa8DAdIGEmmXMOLW93hnDNgzmm9p"
MODEL_RAW_PATH = "model_raw.keras"
MODEL_PATCHED_PATH = "inception_multiscale_fixed.keras"

def strip_quantization_keys(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for k, v in list(obj.items()):
            strip_quantization_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_quantization_keys(item)

def patch_keras_zip(src_zip, dst_zip):
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
# 4. BARRA LATERAL (CONFIGURAÇÕES E PARÂMETROS)
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
st.sidebar.subheader("🎨 Ajustes do Grad-CAM")

cam_sensitivity = st.sidebar.slider(
    "Sensibilidade do Mapa:",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.05,
    help="Valores menores exibem ativações mais sutis e difusas do tumor."
)

cam_opacity = st.sidebar.slider(
    "Opacidade Térmica:",
    min_value=0.30,
    max_value=0.90,
    value=0.70,
    step=0.05,
    help="Define o quão vivo o mapa de calor será sobreposto à lâmina."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Métricas de Validação Independente")
st.sidebar.write("- **Acurácia Global:** 94,81%")
st.sidebar.write("- **ROC AUC:** 0,9893")
st.sidebar.write("- **Sensibilidade:** 91,38%")
st.sidebar.write("- **Especificidade:** 98,96%")

# -------------------------------------------------------------
# 5. ÁREA PRINCIPAL
# -------------------------------------------------------------
st.title("Sistema de Auxílio ao Diagnóstico Histopatológico (H&E)")
st.markdown("Plataforma computacional para classificação e explicabilidade visual de lâminas teciduais.")

# Formulário de Anamnese / Identificação
with st.expander("📋 Identificação da Amostra e Paciente (Opcional)", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        paciente_nome = st.text_input("Nome do Paciente:", placeholder="Ex: Mel, Thor, Fred")
        tutor_nome = st.text_input("Nome do Tutor:", placeholder="Ex: Maria Silva")
    with col_p2:
        especie = st.selectbox("Espécie:", ["Canina", "Felina", "Outra"])
        raca = st.text_input("Raça:", placeholder="Ex: SRD, Poodle, Pastor Alemão")
    with col_p3:
        topografia = st.text_input("Topografia / Órgão de Coleta:", placeholder="Ex: Mama M4, Pele dorsal")
        obs_clinica = st.text_area("Observações Clínicas:", placeholder="Ex: Nódulo firme, ulcerado, evolução rápida", height=68)

info_paciente = {
    "nome": paciente_nome,
    "tutor": tutor_nome,
    "especie": especie,
    "raca": raca,
    "topografia": topografia,
    "obs": obs_clinica
}

uploaded_file = st.file_uploader(
    "Envie a imagem histopatológica para triagem...", 
    type=["jpg", "png", "jpeg", "tif", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size
    superimposed = None

    col1, col2, col3 = st.columns([1, 1, 1])

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
            # Seleciona a última camada convolucional rica em características
            target_layer = model.get_layer("mixed7")
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_array, training=False)
                # Alvo do gradiente: predição de malignidade
                target_class = preds[:, 0]

            # Gradientes dos mapas de ativação em relação à predição
            grads = tape.gradient(target_class, conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Ponderação dos canais convolucionais e aplicação de ReLU
            heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap).numpy()
            heatmap = np.maximum(heatmap, 0)
            # Remove artefatos de zero-padding das bordas do tensor
            heatmap[0, :] = 0
            heatmap[-1, :] = 0
            heatmap[:, 0] = 0
            heatmap[:, -1] = 0

            # Normalização de contraste para destacar focos reais
            if np.max(heatmap) > 1e-7:
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                heatmap = heatmap * prob
            else:
                heatmap = np.zeros_like(heatmap)

            # Redimensionamento suave para o tamanho nativo da imagem
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = np.clip(heatmap_resized, 0, 1)

            # Aplicação do limiar de corte dinâmico da barra lateral
            # Zera ativações frias/de fundo para deixar o tecido limpo
            mask = heatmap_resized >= cam_sensitivity
            heatmap_masked = np.where(mask, heatmap_resized, 0.0)

            # Gerar mapa térmico JET
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_masked), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # Sobreposição direta apenas onde houve ativação
            alpha = np.expand_dims(heatmap_masked * cam_opacity, axis=-1)
            orig_img_np = np.array(image, dtype=np.float32)
            heatmap_color_float = heatmap_color.astype(np.float32)

            blend = orig_img_np * (1.0 - alpha) + heatmap_color_float * alpha
            superimposed = np.clip(blend, 0, 255).astype(np.uint8)

            st.image(superimposed, use_container_width=True)
            st.caption("Focos quentes (amarelo/vermelho) indicam áreas determinantes para a classificação.")
        except Exception as e:
            st.warning(f"Grad-CAM não pôde ser gerado para esta camada: {e}")

    # Geração e Download do Relatório
    st.markdown("---")
    pdf_bytes = generate_pdf_report(
        orig_img_pil=image,
        cam_img_np=superimposed,
        filename=uploaded_file.name,
        prob=prob,
        threshold=threshold,
        diagnostico=diagnostico,
        info_paciente=info_paciente
    )

    clean_name = os.path.splitext(uploaded_file.name)[0]
    st.download_button(
        label="📄 Baixar Laudo Diagnóstico Completo em PDF",
        data=pdf_bytes,
        file_name=f"laudo_{clean_name}.pdf",
        mime="application/pdf",
        use_container_width=True
    )