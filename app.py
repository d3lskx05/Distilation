# app.py — Synonym Checker (+) с загрузкой модели с Google Drive
import os
import tempfile
import streamlit as st
from sentence_transformers import SentenceTransformer
import gdown
import zipfile
import shutil

st.set_page_config(page_title="Synonym Checker", layout="wide")
st.title("🔎 Synonym Checker")

# ======== Сайдбар: настройки модели ========
st.sidebar.header("Настройки модели")
model_source = st.sidebar.selectbox("Источник модели", ["huggingface", "google_drive"], index=0)
DEFAULT_HF = "sentence-transformers/all-MiniLM-L6-v2"

if model_source == "huggingface":
    model_id = st.sidebar.text_input("Hugging Face Model ID", value=DEFAULT_HF)
else:
    model_id = st.sidebar.text_input("Google Drive Folder ID", value="1YHwNG_3O7_k24kx0OB35fe1GUq8kyWz9")

# ======== Функция для загрузки модели с Google Drive ========
@st.cache_resource(show_spinner=True)
def load_model_from_gdrive(folder_id: str):
    """
    Загружает папку модели с Google Drive и возвращает путь к распакованной папке.
    """
    # Создаем временную папку
    tmp_dir = tempfile.mkdtemp()

    # Формируем ссылку для gdown (с экспортом в zip)
    url = f"https://drive.google.com/uc?id={folder_id}"
    zip_path = os.path.join(tmp_dir, "model.zip")

    # Скачиваем zip
    gdown.download(url, zip_path, quiet=False)

    # Распаковываем
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Обычно распаковка создает папку внутри tmp_dir
    # Ищем первую подпапку с config.json
    for root, dirs, files in os.walk(tmp_dir):
        if "config_sentence_transformers.json" in files or "config.json" in files:
            return root

    return tmp_dir  # fallback

# ======== Загрузка модели ========
try:
    with st.spinner("Загружаю модель..."):
        if model_source == "huggingface":
            model = SentenceTransformer(model_id)
        else:
            model_path = load_model_from_gdrive(model_id)
            model = SentenceTransformer(model_path)
    st.sidebar.success("Модель загружена")
except Exception as e:
    st.sidebar.error(f"Не удалось загрузить модель: {e}")
    st.stop()

st.write("Модель готова к использованию!")
