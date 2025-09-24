import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import gdown
import tempfile
import os

st.set_page_config(page_title="Student Distilled Model Demo", layout="wide")
st.title("🔎 Student Distilled Model Demo")

# ======== Sidebar: настройки модели ========
model_source = st.sidebar.selectbox("Источник модели", ["Local path", "Google Drive File ID"])
batch_size = st.sidebar.number_input("Batch size для энкодинга", min_value=8, max_value=512, value=64)

if model_source == "Local path":
    model_path = st.sidebar.text_input("Путь к модели", value="/content/drive/MyDrive/teacher_embs/student_distilled")
else:
    model_gdrive_id = st.sidebar.text_input("1YHwNG_3O7_k24kx0OB35fe1GUq8kyWz9")
    model_path = None  # заполним после загрузки

# ======== Функция загрузки модели ========
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    return SentenceTransformer(path)

def download_from_gdrive(file_id: str):
    tmp_dir = tempfile.mkdtemp()
    dest_path = os.path.join(tmp_dir, "student_model.zip")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", dest_path, quiet=False)
    return tmp_dir, dest_path

# ======== Загрузка модели ========
try:
    if model_source == "Local path":
        model = load_model(model_path)
    else:
        tmp_dir, zip_path = download_from_gdrive(model_gdrive_id)
        # Распаковка zip
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)
        model = load_model(tmp_dir)
    st.sidebar.success("Модель загружена ✅")
except Exception as e:
    st.sidebar.error(f"Ошибка при загрузке модели: {e}")
    st.stop()

# ======== Основной ввод ========
st.header("Encode Sentences")
text_input = st.text_area("Введите предложения (по одному на строку):")

if st.button("Encode & Show Embeddings"):
    sentences = [line.strip() for line in text_input.split("\n") if line.strip()]
    if sentences:
        embeddings = model.encode(sentences, convert_to_tensor=True, batch_size=batch_size)
        st.write("Эмбеддинги (первые 5 векторов):")
        st.write(embeddings[:5])

        if len(sentences) > 1:
            sim_matrix = util.cos_sim(embeddings, embeddings)
            st.write("Cosine similarity матрица:")
            st.write(sim_matrix)
