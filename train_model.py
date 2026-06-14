import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# TIMER
# ==================================================

start_time = time.time()

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")

df = pd.read_csv("data/GoEmotions.csv")

print("=" * 60)
print("INFORMASI DATASET")
print("=" * 60)

print("\nShape Dataset:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

# ==================================================
# MENENTUKAN KOLOM LABEL
# ==================================================

emotion_columns = df.columns[9:]

print("\nJumlah Label Emosi:")
print(len(emotion_columns))

print("\nNama Label Emosi:")
print(emotion_columns.tolist())

# ==================================================
# DISTRIBUSI LABEL
# ==================================================

print("\nMenghitung distribusi label...")

label_counts = (
    df[emotion_columns]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Label:")

print(label_counts.head(10))

# ==================================================
# SIMPAN GRAFIK DISTRIBUSI LABEL
# ==================================================

plt.figure(figsize=(14, 6))

label_counts.plot(kind="bar")

plt.title("Distribusi Label Emosi GoEmotions")
plt.xlabel("Label")
plt.ylabel("Jumlah Data")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "distribusi_label.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGrafik disimpan:")
print("distribusi_label.png")

# ==================================================
# ANALISIS MULTI LABEL
# ==================================================

label_per_text = df[emotion_columns].sum(axis=1)

print("\nRata-rata jumlah label per teks:")
print(round(label_per_text.mean(), 2))

# ==================================================
# MEMBUAT LABEL UTAMA
# ==================================================

print("\nMembuat label utama...")

df["label"] = df[emotion_columns].idxmax(axis=1)

print("Label utama berhasil dibuat.")

print("\nDistribusi Label Utama:")

print(
    df["label"]
    .value_counts()
    .head(10)
)

# ==================================================
# PREPROCESSING TEXT
# ==================================================

print("\nMulai preprocessing...")

def clean_text(text):

    text = str(text).lower()

    # hapus URL
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # hapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # hapus tanda baca
    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    # hapus spasi berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

print("Preprocessing selesai.")

print("\nContoh Data:")

print(
    df[
        ["text", "clean_text"]
    ].head()
)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

print("\nMelakukan train-test split...")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Jumlah Data Train :", len(X_train))
print("Jumlah Data Test  :", len(X_test))

# ==================================================
# TF-IDF
# ==================================================

print("\nMembangun TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)

print("TF-IDF selesai.")

print("\nShape Train:")
print(X_train_tfidf.shape)

print("\nShape Test:")
print(X_test_tfidf.shape)

# ==================================================
# TRAINING MODEL
# ==================================================

print("\nMelatih Logistic Regression...")

model = LogisticRegression(
    max_iter=300,
    solver="liblinear",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Training selesai.")

# ==================================================
# EVALUASI
# ==================================================

print("\nMelakukan evaluasi...")

y_pred = model.predict(
    X_test_tfidf
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix berhasil dibuat.")

# ==================================================
# SIMPAN MODEL
# ==================================================

print("\nMenyimpan model...")

joblib.dump(
    model,
    "models/emotion_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("Model berhasil disimpan.")

print("\nFile yang dibuat:")
print("models/emotion_model.pkl")
print("models/tfidf_vectorizer.pkl")
print("distribusi_label.png")

# ==================================================
# TOTAL WAKTU
# ==================================================

end_time = time.time()

print(
    f"\nTotal waktu eksekusi: {round(end_time - start_time, 2)} detik"
)