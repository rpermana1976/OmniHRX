import os
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import Word

# Download NLTK resources jika belum ada
nltk.download('punkt')
nltk.download('stopwords')

# Buat stemmer Bahasa Indonesia
factory = StemmerFactory()
indonesian_stemmer = factory.create_stemmer()

# Load stopwords untuk Bahasa Inggris dan Indonesia
stop_words_eng = set(stopwords.words('english'))
stop_words_ind = set(stopwords.words('indonesian'))

def detect_language(text):
    """Deteksi bahasa teks (Indonesia atau Inggris)."""
    try:
        return detect(text)
    except:
        return "unknown"

def clean_text(text):
    """Bersihkan teks dari karakter tidak perlu."""
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def preprocess_text(text):
    """Preprocessing teks: cleaning, tokenization, stopword removal, stemming/lemmatization."""
    text = clean_text(text)
    lang = detect_language(text)

    tokens = word_tokenize(text)

    if lang == "id":
        tokens = [indonesian_stemmer.stem(word) for word in tokens if word not in stop_words_ind]
    elif lang == "en":
        tokens = [Word(word).lemmatize() for word in tokens if word not in stop_words_eng]
    
    return " ".join(tokens)

def process_folder(input_folder, output_folder):
    """Memproses semua file teks dalam folder dan menyimpan hasil preprocessing."""
    if not os.path.exists(input_folder):
        print(f"⚠️ Folder tidak ditemukan: {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_file_path):
            print(f"📄 Memproses: {filename}")
            with open(input_file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            processed_text = preprocess_text(text)

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(processed_text)
            
            print(f"✅ Preprocessed text disimpan di: {output_file_path}")

if __name__ == "__main__":
    print("🚀 Memulai preprocessing teks...")
    process_folder("data/processed_data/resumes", "data/final_data/resumes")
    process_folder("data/processed_data/jobs", "data/final_data/jobs")
    print("🎉 Preprocessing selesai!")
