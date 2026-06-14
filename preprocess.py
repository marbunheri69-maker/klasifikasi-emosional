import re
import string

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # 1. Case Folding
    text = text.lower()

    # 2. Hapus URL
    text = re.sub(
        r"http\S+",
        "",
        text
    )

    # 3. Hapus Angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # 4. Hapus Tanda Baca
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # 5. Tokenisasi
    tokens = word_tokenize(text)

    # 6. Lemmatization
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    # Gabungkan kembali
    text = " ".join(tokens)

    return text