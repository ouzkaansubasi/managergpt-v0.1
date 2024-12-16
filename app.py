import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import speech_recognition as sr
import soundfile as sf
from io import BytesIO
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import fitz  # PyMuPDF
import sounddevice as sd
import wavio
import pyttsx3
from Foundation import NSObject
from AppKit import NSSpeechSynthesizer
import sqlite3
import hashlib
from streamlit_lottie import st_lottie
import requests
import json
import duckdb
import random

# CSS for styling
day_mode_css = """
<style>
body {
    background-color: #d3d3d3; /* Açık gri arka plan */
    color: #1c1c1e;
    font-family: 'Arial', sans-serif;
}
.main {
    background-color: #f5f5f5; /* Daha açık bir gri arka plan */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.sidebar .sidebar-content {
    background-color: #2c2c2e;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}
.sidebar .sidebar-content:hover {
    background-color: #3a3a3c;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
}
.sidebar .sidebar-content h2 {
    color: #f5f5f5;
    font-size: 1.5em;
    border-bottom: 1px solid #4a4a4c;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.sidebar .sidebar-content .element-container {
    color: #f5f5f5;
    margin-bottom: 20px;
}
.sidebar .sidebar-content .element-container .element {
    background-color: #3a3a3c;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}
.sidebar .sidebar-content .element-container .element:hover {
    background-color: #4a4a4c;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.sidebar .sidebar-content .element-container .element img {
    width: 100%;
    border-radius: 5px;
    margin-bottom: 10px;
    border: 2px solid red;
}
.stTextInput label {
    color: #1c1c1e; /* Değiştirilmiş renk */
}
.stTextInput input {
    background-color: #3a3a3c;
    color: #f5f5f5;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
}
.stButton button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    transition: background-color 0.3s, transform 0.3s;
}
.stButton button:hover {
    background-color: #0056b3;
    color: white;
    transform: translateY(-2px);
}
.stFileUploader label {
    color: #1c1c1e; /* Değiştirilmiş renk */
}
.stFileUploader div div div {
    background-color: #3a3a3c;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
}
.chat-container {
    background-color: #3a3a3c;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    margin-top: 20px;
}
.chat-container h2 {
    color: #f5f5f5;
}
.chat-message {
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 10px;
}
.chat-message.user {
    background-color: #0056b3;
    color: white;
    text-align: right;
}
.chat-message.assistant {
    background-color: #4a4a4c;
    color: white;
}
.hyperparameter-expander .st-expander-header {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content {
    background-color: #4a4a4c;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.hyperparameter-expander .st-expander-content label {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content .st-slider {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content .st-selectbox {
    color: #f5f5f5;
}
.pdf-preview {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 20px;
}
.pdf-preview img {
    max-width: 100%;
    border: 1px solid #fff;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.title {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 20px;
    color: #007bff;
    text-align: center;
}
.footer {
    text-align: center;
    margin-top: 50px;
    color: #bbb;
}
.login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 40px;
    background-color: #2c2c2e;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
}
.login-container h1 {
    margin-bottom: 20px;
    color: #007bff;
}
.login-container input {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
    background-color: #3a3a3c;
    color: #f5f5f5;
}
.login-container button {
    width: 100%;
    padding: 10px;
    border: none;
    border-radius: 5px;
    background-color: #007bff;
    color: white;
    font-size: 16px;
}
.login-container button:hover {
    background-color: #0056b3;
}
</style>
"""

dark_mode_css = """
<style>
body {
    background-color: #1e1e1e; /* Koyu arka plan */
    color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}
.main {
    background-color: #2e2e2e; /* Daha koyu bir gri arka plan */
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.sidebar .sidebar-content {
    background-color: #1c1c1e;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}
.sidebar .sidebar-content:hover {
    background-color: #2c2c2e;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
}
.sidebar .sidebar-content h2 {
    color: #ffffff;
    font-size: 1.5em;
    border-bottom: 1px solid #4a4a4c;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
.sidebar .sidebar-content .element-container {
    color: #f5f5f5;
    margin-bottom: 20px;
}
.sidebar .sidebar-content .element-container .element {
    background-color: #2c2c2e;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    transition: all 0.3s ease;
}
.sidebar .sidebar-content .element-container .element:hover {
    background-color: #3a3a3c;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.sidebar .sidebar-content .element-container .element img {
    width: 100%;
    border-radius: 5px;
    margin-bottom: 10px;
    border: 2px solid red;
}
.stTextInput label {
    color: #f5f5f5; /* Değiştirilmiş renk */
}
.stTextInput input {
    background-color: #3a3a3c;
    color: #f5f5f5;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
}
.stButton button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    transition: background-color 0.3s, transform 0.3s;
}
.stButton button:hover {
    background-color: #0056b3;
    color: white;
    transform: translateY(-2px);
}
.stFileUploader label {
    color: #f5f5f5; /* Değiştirilmiş renk */
}
.stFileUploader div div div {
    background-color: #3a3a3c;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
}
.chat-container {
    background-color: #2c2c2e;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    margin-top: 20px;
}
.chat-container h2 {
    color: #ffffff;
}
.chat-message {
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 10px;
}
.chat-message.user {
    background-color: #0056b3;
    color: white;
    text-align: right;
}
.chat-message.assistant {
    background-color: #4a4a4c;
    color: white;
}
.hyperparameter-expander .st-expander-header {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content {
    background-color: #4a4a4c;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.hyperparameter-expander .st-expander-content label {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content .st-slider {
    color: #f5f5f5;
}
.hyperparameter-expander .st-expander-content .st-selectbox {
    color: #f5f5f5;
}
.pdf-preview {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 20px;
}
.pdf-preview img {
    max-width: 100%;
    border: 1px solid #fff;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
}
.title {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 20px;
    color: #007bff;
    text-align: center;
}
.footer {
    text-align: center;
    margin-top: 50px;
    color: #bbb;
}
.login-container {
    max-width: 400px;
    margin: 0 auto;
    padding: 40px;
    background-color: #1c1c1e;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
}
.login-container h1 {
    margin-bottom: 20px;
    color: #ffffff;
}
.login-container input {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border: 1px solid #4a4a4c;
    border-radius: 5px;
    background-color: #3a3a3c;
    color: #f5f5f5;
}
.login-container button {
    width: 100%;
    padding: 10px;
    border: none;
    border-radius: 5px;
    background-color: #007bff;
    color: white;
    font-size: 16px;
}
.login-container button:hover {
    background-color: #0056b3;
}
</style>
"""

def apply_css(dark_mode):
    if dark_mode:
        st.markdown(dark_mode_css, unsafe_allow_html=True)
    else:
        st.markdown(day_mode_css, unsafe_allow_html=True)

# Başlangıçta gün modu
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# CSS uygulama
apply_css(st.session_state.dark_mode)

# Karanlık mod butonu
if st.button('Karanlık Modu Aç/Kapat'):
    st.session_state.dark_mode = not st.session_state.dark_mode
    apply_css(st.session_state.dark_mode)
    st.experimental_rerun()

# Veritabanı bağlantısı
conn = sqlite3.connect('database.db')
c = conn.cursor()

# Tablo oluşturma
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT,
        response TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')

conn.commit()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Kullanıcı girişi ve kayıt işlemleri
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register(username, password):
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login(username, password):
    c.execute('SELECT id, username FROM users WHERE username = ? AND password = ?', (username, hash_password(password)))
    return c.fetchone()

# Sidebar içerikleri
with st.sidebar:
    st.image("logo.png", use_column_width=True, width=250)
    st.title("Sanal Asistan")
    st.markdown("""
    **Proje Tanıtımı:**

    Bu proje, yazılım proje yöneticileri için geliştirilmiş bir sanal asistandır. 
    Proje kapsamında PDF, Excel ve ses dosyalarını analiz edebilme yeteneğine sahip olan bu sanal asistan, 
    doğal dil işleme ve makine öğrenimi tekniklerini kullanarak kullanıcı sorularına akıllı cevaplar verebilmektedir.
    
    **Proje Özellikleri:**
    - PDF dosya analizi ve soru-cevap işlemleri
    - Excel dosya analizi ve görselleştirme
    - Ses dosyası transkripsiyonu ve analizi

    Projenin amacı, yazılım proje yöneticilerinin işlerini kolaylaştırmak ve verimliliklerini artırmaktır.
    """)
    st.write("Oğuz Kaan SUBAŞI - 191180076")
    st.write("Selin Cansu AKBAŞ - 191180005")

    json2 = load_lottiefile("/Users/oguzkaansubasi/Desktop/managergpt-v0.1/ani2.json")

    st_lottie(
        json2,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height = 250,
        width=250,
        key=None,
    )

    load_dotenv()

    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
        st.session_state['username'] = ""

def record_audio(filename, duration, fs, channels):
    """Ses kaydı yap ve bir WAV dosyasına kaydet."""
    st.write("Kaydediliyor...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Kayıt bitene kadar bekle
    wavio.write(filename, recording, fs, sampwidth=2)
    st.write("Kayıt tamamlandı.")

def convert_pdf_to_image(pdf_path):
    """Convert the first page of a PDF to an image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)  # İlk sayfa
    pix = page.get_pixmap()
    img_path = "cover.png"
    pix.save(img_path)
    return img_path

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    audio_format = audio_file.name.split('.')[-1].lower()
    if audio_format == 'mp3':
        audio_data, samplerate = sf.read(BytesIO(audio_file.read()))
        audio_file = "converted_audio.wav"
        sf.write(audio_file, audio_data, samplerate)
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return "Üzgünüm, ses dosyasını anlayamadım."
    except sr.RequestError as e:
        return f"Google Speech Recognition hizmetine ulaşılamadı; {e}"

def main():
    if st.session_state['user_id'] is None:
        
        st.markdown('<h1>Giriş Yap veya Kayıt Ol</h1>', unsafe_allow_html=True)

        choice = st.selectbox("Seçiminizi yapın", ["Giriş Yap", "Kayıt Ol"])

        if choice == "Giriş Yap":
            username = st.text_input("Kullanıcı Adı", key="login_username")
            password = st.text_input("Şifre", type="password", key="login_password")
            if st.button("Giriş Yap"):
                user = login(username, password)
                if user:
                    st.session_state['user_id'] = user[0]
                    st.session_state['username'] = user[1]
                    st.success("Giriş başarılı!")
                    st.experimental_rerun()
                else:
                    st.error("Kullanıcı adı veya şifre yanlış.")

        if choice == "Kayıt Ol":
            username = st.text_input("Kullanıcı Adı", key="register_username")
            password = st.text_input("Şifre", type="password", key="register_password")
            if st.button("Kayıt Ol"):
                if register(username, password):
                    st.success("Kayıt başarılı! Lütfen giriş yapın.")
                else:
                    st.error("Bu kullanıcı adı zaten alınmış.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.write(f"Hoşgeldiniz, {st.session_state['username']}!")
        if st.button("Çıkış Yap"):
            st.session_state['user_id'] = None
            st.session_state['username'] = ""
            st.experimental_rerun()
        app_choice = st.selectbox("Seçiminizi yapın", ["PDF Analizi", "Excel Analizi", "Ses Analizi"])
        if app_choice == "PDF Analizi":
            run_pdf_app()
        elif app_choice == "Excel Analizi":
            run_excel_app()
        elif app_choice == "Ses Analizi":
            run_audio_app()

def run_pdf_app():
    st.markdown('<div class="title">PDF Dosyası ile Sohbet Başlat</div>', unsafe_allow_html=True)

    with st.expander("Hiperparametre Ayarları", expanded=True):
        st.markdown("""
            <div class="hyperparameter-expander">
            """, unsafe_allow_html=True)
        
        # Hiperparametre ayarları
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        k_value = st.slider("Top K Similarity Search Results", 1, 10, 3)
        model_name = st.selectbox("Model Name", ['gpt-3.5-turbo', 'gpt-3', 'gpt-4'])

        st.markdown("""
            </div>
            """, unsafe_allow_html=True)

    # PDF dosyası yükleme
    pdf = st.file_uploader("PDF dosyanızı yükleyin", type='pdf')

    if pdf is not None:
        # PDF dosyasını geçici bir dosyaya kaydet
        pdf_path = f"uploaded_{pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf.read())

        # PDF kapak resmini oluşturma
        cover_image = convert_pdf_to_image(pdf_path)

        # PDF önizlemesi
        st.markdown("<div class='pdf-preview'>", unsafe_allow_html=True)
        st.subheader("PDF Kapak Resmi")
        st.image(cover_image)
        st.markdown("</div>", unsafe_allow_html=True)

        pdf_reader = PdfReader(pdf_path)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        # Embeddings

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("PDF dosyası hakkında sorular sor: ")

        json1 = load_lottiefile("/Users/oguzkaansubasi/Desktop/managergpt-v0.1/ani1.json")

        st_lottie(
            json1,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height = 250,
            width=250,
            key=None,
        )
        
        # Sesli soru sorma
        if st.button("Sesli Soru Sor"):
            duration = 5  # seconds
            fs = 44100  # Sample rate
            channels = 1  # Mono channel
            filename = "output.wav"

            record_audio(filename, duration, fs, channels)

            recognizer = sr.Recognizer()
            with sr.AudioFile(filename) as source:
                audio = recognizer.record(source)

                try:
                    query = recognizer.recognize_google(audio, language="tr-TR")
                    st.write(f"Sorunuz: {query}")
                except sr.UnknownValueError:
                    st.write("Üzgünüm, ne dediğinizi anlayamadım.")
                except sr.RequestError:
                    st.write("Google Speech Recognition hizmetine ulaşılamadı.")

        response = ""
        if query:
            docs = VectorStore.similarity_search(query=query, k=k_value)
            
            llm = OpenAI(model_name=model_name)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Sohbet geçmişini güncelle
            if "chat_history_pdf" not in st.session_state:
                st.session_state["chat_history_pdf"] = []

            st.session_state["chat_history_pdf"].append({"role": "user", "content": query})
            st.session_state["chat_history_pdf"].append({"role": "assistant", "content": response})

            # Sohbet ekranını güncelle
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for chat in st.session_state["chat_history_pdf"]:
                role = chat["role"]
                content = chat["content"]
                css_class = "chat-message user" if role == "user" else "chat-message assistant"
                st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state["response_pdf"] = response

        if "response_pdf" in st.session_state:
            response = st.session_state["response_pdf"]
            if st.button("Cevabı Sesli Olarak Al"):
                # NSSpeechSynthesizer kullanarak metni sese dönüştürme
                synthesizer = NSSpeechSynthesizer.alloc().init()
                synthesizer.startSpeakingString_(response)
                st.write(f"Cevap: {response}")

    st.markdown('<div class="footer">© 2024 Sanal Asistan. Tüm hakları saklıdır.</div>', unsafe_allow_html=True)

def run_excel_app():
    st.markdown('<div class="title">Excel Dosyası Analizi</div>', unsafe_allow_html=True)

    excel_file = st.file_uploader("Excel dosyanızı yükleyin", type=['xlsx', 'xls'])
    
    if excel_file is not None:
        df = pd.read_excel(excel_file)

        st.subheader("Excel Verileri")
        st.dataframe(df)

        # Grafik oluşturma
        st.subheader("Veri Grafiği")
        columns = df.columns.tolist()
        x_axis = st.selectbox("X Ekseni Seçin", columns)
        y_axis = st.selectbox("Y Ekseni Seçin", columns)

        if st.button("Grafik Oluştur"):
            fig, ax = plt.subplots()
            df.plot(kind='bar', x=x_axis, y=y_axis, ax=ax)
            st.pyplot(fig)

        # Rapor oluşturma
        st.subheader("Rapor")
        report = df.describe().transpose()
        st.write(report)

        # Verilerin analizi ve yorumu
        st.subheader("Analiz ve Yorum")
        analysis_text = analyze_data(df, x_axis, y_axis)
        st.write(analysis_text)

        # Yeni eklenen bölümler
        all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        # Metric and Gauge Graphs
        plot_metric(
            "Total Accounts Receivable",
            6621280,
            prefix="$",
            suffix="",
            show_graph=True,
            color_graph="rgba(0, 104, 201, 0.2)",
        )
        plot_gauge(1.86, "#0068C9", "%", "Current Ratio", 3)
        plot_metric(
            "Total Accounts Payable",
            1630270,
            prefix="$",
            suffix="",
            show_graph=True,
            color_graph="rgba(255, 43, 43, 0.2)",
        )
        plot_gauge(10, "#FF8700", " days", "In Stock", 31)
        plot_metric("Equity Ratio", 75.38, prefix="", suffix=" %", show_graph=False)
        plot_gauge(7, "#FF2B2B", " days", "Out Stock", 31)
        plot_metric("Debt Equity", 1.10, prefix="", suffix=" %", show_graph=False)
        plot_gauge(28, "#29B09D", " days", "Delay", 31)

        # Main Graphs
        plot_top_right(df, all_months)
        plot_bottom_left(df, all_months)
        plot_bottom_right(df, all_months)

    st.markdown('<div class="footer">© 2024 Sanal Asistan. Tüm hakları saklıdır.</div>', unsafe_allow_html=True)

def analyze_data(df, x_axis, y_axis):
    """
    Verileri analiz eden ve yorumlayan bir fonksiyon.
    """
    analysis = f"{x_axis} ve {y_axis} eksenlerindeki verilerin analizi:\n\n"
    analysis += f"{x_axis} ve {y_axis} arasında genel bir ilişki var gibi görünüyor. "
    analysis += f"Örneğin, {y_axis} değeri {x_axis} değeri arttıkça artma eğilimindedir. "
    analysis += f"Bu, {x_axis} değeri yüksek olan ülkelerde {y_axis} değerinin de yüksek olduğu anlamına gelir.\n\n"
    
    analysis += "Verilerin istatistiksel özetine bakıldığında:\n"
    stats = df.describe().transpose()
    for index, row in stats.iterrows():
        analysis += f"{index}:\n"
        analysis += f"  Ortalama: {row['mean']}\n"
        analysis += f"  Std Sapma: {row['std']}\n"
        analysis += f"  Min: {row['min']}\n"
        analysis += f"  25%: {row['25%']}\n"
        analysis += f"  50%: {row['50%']}\n"
        analysis += f"  75%: {row['75%']}\n"
        analysis += f"  Max: {row['max']}\n\n"
    
    analysis += "Bu analiz, {x_axis} ve {y_axis} arasındaki ilişkiyi daha iyi anlamamıza yardımcı olabilir. "
    analysis += "Daha fazla ayrıntılı analiz ve görselleştirme için ek araçlar kullanabilirsiniz."
    
    return analysis

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=[random.randint(0, 100) for _ in range(30)],
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_gauge(indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_top_right(df, all_months):
    sales_data = duckdb.sql(
        f"""
        WITH sales_data AS (
            UNPIVOT ( 
                SELECT 
                    Scenario,
                    business_unit,
                    {','.join(all_months)} 
                    FROM df 
                    WHERE Year='2023' 
                    AND Account='Sales' 
                ) 
            ON {','.join(all_months)}
            INTO
                NAME month
                VALUE sales
        ),

        aggregated_sales AS (
            SELECT
                Scenario,
                business_unit,
                SUM(sales) AS sales
            FROM sales_data
            GROUP BY Scenario, business_unit
        )
        
        SELECT * FROM aggregated_sales
        """
    ).df()

    fig = px.bar(
        sales_data,
        x="business_unit",
        y="sales",
        color="Scenario",
        barmode="group",
        text_auto=".2s",
        title="Sales for Year 2023",
        height=400,
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bottom_left(df, all_months):
    sales_data = duckdb.sql(
        f"""
        WITH sales_data AS (
            SELECT 
            Scenario,{','.join(all_months)} 
            FROM df 
            WHERE Year='2023' 
            AND Account='Sales'
            AND business_unit='Software'
        )

        UNPIVOT sales_data 
        ON {','.join(all_months)}
        INTO
            NAME month
            VALUE sales
    """
    ).df()

    fig = px.line(
        sales_data,
        x="month",
        y="sales",
        color="Scenario",
        markers=True,
        text="sales",
        title="Monthly Budget vs Forecast 2023",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

def plot_bottom_right(df, all_months):
    sales_data = duckdb.sql(
        f"""
        WITH sales_data AS (
            UNPIVOT ( 
                SELECT 
                    Account,Year,{','.join([f'ABS({month}) AS {month}' for month in all_months])}
                    FROM df 
                    WHERE Scenario='Actuals'
                    AND Account!='Sales'
                ) 
            ON {','.join(all_months)}
            INTO
                NAME year
                VALUE sales
        ),

        aggregated_sales AS (
            SELECT
                Account,
                Year,
                SUM(sales) AS sales
            FROM sales_data
            GROUP BY Account, Year
        )
        
        SELECT * FROM aggregated_sales
    """
    ).df()

    fig = px.bar(
        sales_data,
        x="Year",
        y="sales",
        color="Account",
        title="Actual Yearly Sales Per Account",
    )
    st.plotly_chart(fig, use_container_width=True)

def run_audio_app():
    st.markdown('<div class="title">Ses Dosyası Analizi</div>', unsafe_allow_html=True)

    audio_file = st.file_uploader("Ses dosyanızı yükleyin", type=['wav', 'mp3'])

    if audio_file is not None:
        st.audio(audio_file)
        st.subheader("Ses Dosyası Metni")
        text = transcribe_audio(audio_file)
        st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        query = st.text_input("Ses dosyası hakkında sorular sor: ")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            
            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Sohbet geçmişini güncelle
            if "chat_history_audio" not in st.session_state:
                st.session_state["chat_history_audio"] = []

            st.session_state["chat_history_audio"].append({"role": "user", "content": query})
            st.session_state["chat_history_audio"].append({"role": "assistant", "content": response})

            # Sohbet ekranını güncelle
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for chat in st.session_state["chat_history_audio"]:
                role = chat["role"]
                content = chat["content"]
                css_class = "chat-message user" if role == "user" else "chat-message assistant"
                st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.session_state["response_audio"] = response

        if "response_audio" in st.session_state:
            response = st.session_state["response_audio"]
            if st.button("Cevabı Sesli Olarak Al"):
                # NSSpeechSynthesizer kullanarak metni sese dönüştürme
                synthesizer = NSSpeechSynthesizer.alloc().init()
                synthesizer.startSpeakingString_(response)
                st.write(f"Cevap: {response}")

    st.markdown('<div class="footer">© 2024 Sanal Asistan. Tüm hakları saklıdır.</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
