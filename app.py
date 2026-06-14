import streamlit as st
import joblib
import re
import string
import pandas as pd
import altair as alt

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "models/emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# ==================================================
# PREPROCESSING
# ==================================================

def clean_text(text):

    text = str(text).lower()

    # Hapus URL
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # Hapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Hapus tanda baca
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ==================================================
# EMOJI UNTUK LABEL
# ==================================================

emotion_visuals = {
    "admiration": "👏",
    "amusement": "😂",
    "anger": "😠",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "😍",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😭",
    "joy": "😄",
    "love": "🥰",
    "nervousness": "😬",
    "neutral": "😐",
    "optimism": "🌞",
    "pride": "🏆",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😢",
    "surprise": "😲"
}

def get_visual(emotion):
    return emotion_visuals.get(
        emotion.lower(),
        "😶"
    )

# ==================================================
# CUSTOM CSS (Background Biru & Tombol Hijau)
# ==================================================

st.markdown("""
<style>
/* Background Biru Gradasi Gelap */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Memastikan semua teks utama berwarna putih agar terbaca */
h1, h2, h3, .stMarkdown, .stWrite, label, p {
    color: white !important;
}

/* Tombol Hijau Terang agar kontras dengan background biru */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: bold;
    padding: 12px;
    border: none;
    color: white;
    background: linear-gradient(90deg, #11998e, #38ef7d);
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #0d7a71, #2cb863);
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

/* Kotak Input Teks */
.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #38ef7d;
}

.emoji-style {
    font-size: 100px;
    text-align: center;
    margin-top: 20px;
}

.result-style {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #38ef7d;
    text-transform: uppercase;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("😊 Emotion Classification System")

st.write(
    "Masukkan sebuah teks dan sistem akan memprediksi emosi yang terkandung di dalamnya menggunakan model Machine Learning (gunakan kalimat berbahasa inggris)."
)

# ==================================================
# INPUT USER
# ==================================================

user_input = st.text_area(
    "Masukkan Teks:",
    placeholder="Contoh: I am very happy today!",
    height=150
)

# ==================================================
# PREDIKSI & DIAGRAM
# ==================================================

if st.button("🔍 Analisis Emosi"):

    if user_input.strip() == "":

        st.warning(
            "Masukkan teks terlebih dahulu."
        )

    else:

        # Preprocessing
        cleaned_text = clean_text(
            user_input
        )

        # TF-IDF
        vectorized_text = vectorizer.transform(
            [cleaned_text]
        )

        # Prediksi Label Utama
        prediction = model.predict(
            vectorized_text
        )[0]

        # Ambil Emoji
        emoji = get_visual(
            prediction
        )

        # Tampilkan Hasil Visual Utama
        st.markdown(
            f"<div class='emoji-style'>{emoji}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='result-style'>{prediction}</div>",
            unsafe_allow_html=True
        )

        st.success(
            f"Emosi yang terdeteksi adalah: {prediction}"
        )

       # ==================================================
        # DIAGRAM PERBANDINGAN (DENGAN WARNA CUSTOM)
        # ==================================================
        st.write("---")
        st.subheader("📊 Perbandingan Prediksi Emosi")
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized_text)[0]
            classes = model.classes_
            
            chart_data = pd.DataFrame({
                "Emosi": classes,
                "Keyakinan": proba * 100
            })
            
            chart_data = chart_data.sort_values(by="Keyakinan", ascending=False).head(5)
            
           # Membuat diagram dengan warna yang serasi
            chart = alt.Chart(chart_data).mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3,
                color='#38ef7d' 
            ).encode(
                x=alt.X('Emosi', sort='-y', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('Keyakinan', title='Keyakinan (%)', axis=alt.Axis(labelColor='white', titleColor='white')),
                tooltip=['Emosi', 'Keyakinan']
            ).properties(
                height=300,
                background='transparent' # Pastikan background chart transparan
            ).configure_axis(
                gridColor='#4f6d7a', # Warna garis bantu yang lebih gelap agar tidak mengganggu
                domainColor='white'
            ).configure_view(
                stroke='transparent' # Menghilangkan garis bingkai grafik
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            st.write("Sistem menampilkan 5 emosi dengan skor keyakinan tertinggi.")
        else:
            st.info("Model ini tidak mendukung visualisasi probabilitas.")