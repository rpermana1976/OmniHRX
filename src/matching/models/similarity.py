import os
import sys
import numpy as np
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

# Pastikan Python mengenali folder `src`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 🔥 **Gunakan GPU untuk SBERT jika tersedia**
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2', device=device)

# 🔥 **Load Pretrained Word2Vec & FastText**
word2vec_model = KeyedVectors.load_word2vec_format("models/pretrained/GoogleNews-vectors-negative300.bin", binary=True)
fasttext_model = KeyedVectors.load_word2vec_format("models/pretrained/cc.id.300.vec", binary=False)

# 🔥 **PCA untuk Reduksi Dimensi**
def reduce_dimensionality(model, n_components=100):
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(model.vectors)
    model.vectors = reduced_vectors
    return model

word2vec_model = reduce_dimensionality(word2vec_model)
fasttext_model = reduce_dimensionality(fasttext_model)

# 🔥 **Cache Embedding**
def save_embedding_cache(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_embedding_cache(filename):
    """Load cache jika ada, return dictionary kosong jika tidak ada."""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return {}

# 🔥 **Fungsi untuk mendapatkan vektor kalimat**
def get_sentence_vector(sentence, model, cache_filename):
    """Mengubah kalimat menjadi vektor menggunakan Word2Vec atau FastText, dengan cache dan normalisasi."""
    cache = load_embedding_cache(cache_filename) or {}
    if sentence in cache:
        return cache[sentence]

    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(model.vector_size)

    sentence_vector = np.mean(word_vectors, axis=0)
    sentence_vector = normalize(sentence_vector.reshape(1, -1))[0]
    
    # Simpan ke cache
    cache[sentence] = sentence_vector
    save_embedding_cache(cache, cache_filename)
    return sentence_vector

# 🔥 **Fungsi untuk menghitung kesamaan teks menggunakan SBERT**
def sbert_similarity(text1, text2):
    emb1 = sbert_model.encode(text1, convert_to_tensor=True)
    emb2 = sbert_model.encode(text2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

# 📌 **Fungsi utama untuk mencocokkan CV dengan job descriptions**
def match_resume_to_jobs(resume_text, job_texts):
    scores = []
    detailed_results = []

    for job_text in job_texts:
        w2v_score = np.dot(get_sentence_vector(resume_text, word2vec_model, "cache/w2v.pkl"),
                           get_sentence_vector(job_text, word2vec_model, "cache/w2v.pkl"))
        ft_score = np.dot(get_sentence_vector(resume_text, fasttext_model, "cache/ft.pkl"),
                          get_sentence_vector(job_text, fasttext_model, "cache/ft.pkl"))
        sbert_score = sbert_similarity(resume_text, job_text)

        # 🔥 **Hitung Final Score dengan bobot masing-masing model**
        final_score = (0.10 * w2v_score) + (0.20 * ft_score) + (0.70 * sbert_score)

        # 🔥 Simpan detail setiap skor
        detailed_results.append({
            "job_text": job_text,
            "w2v_score": round(w2v_score, 4),
            "ft_score": round(ft_score, 4),
            "sbert_score": round(sbert_score, 4),
            "final_score": round(final_score, 4)
        })

        scores.append(final_score)

    return scores, detailed_results  

# 🔥 **Parallel Processing dengan ThreadPoolExecutor**
if __name__ == "__main__":
    print("🚀 Memulai pencocokan CV dengan job descriptions...")

    # Load CV & Job Descriptions
    resume_files = os.listdir("data/final_data/resumes")
    job_files = os.listdir("data/final_data/jobs")

    resumes = [open(f"data/final_data/resumes/{file}", "r", encoding="utf-8").read() for file in resume_files]
    jobs = [open(f"data/final_data/jobs/{file}", "r", encoding="utf-8").read() for file in job_files]

    def process_cv(resume):
        return match_resume_to_jobs(resume, jobs)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_cv, resumes))

# 🔥 **Buat DataFrame untuk Heatmap**
cv_names = [cv.replace(".txt", "") for cv in resume_files]  
job_names = [job.replace(".txt", "") for job in job_files]  
similarity_matrix = pd.DataFrame([result[0] for result in results], index=cv_names, columns=job_names)

# 🔥 **Atur threshold similarity untuk highlight warna**
threshold = 0.75  
max_similarity = similarity_matrix.max().max()  
min_similarity = similarity_matrix.min().min()

# 🔥 **Buat mask untuk highlight top matches**
top_match_mask = similarity_matrix.apply(lambda x: x == x.max(), axis=1)

plt.figure(figsize=(12, 8))

# 🔥 **Plot Heatmap dengan skala warna "RdYlGn"**
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5,
            vmin=min_similarity, vmax=max_similarity, cbar_kws={'label': 'Similarity Score'})

# 🔥 **Tambahkan threshold garis**
plt.axhline(y=-0.5, color='black', linewidth=2)  
plt.axvline(x=-0.5, color='black', linewidth=2)

# 🔥 **Judul dan Label**
plt.title("🔥 Heatmap Similarity CV vs Job Descriptions 🔥", fontsize=14, fontweight="bold")
plt.xlabel("Lowongan Kerja", fontsize=12)
plt.ylabel("CV Pelamar", fontsize=12)
plt.xticks(rotation=45, ha="right")  
plt.yticks(fontsize=10)

# 🔥 **Tambahkan legenda keterangan skala warna**
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn"), ax=plt.gca())
cbar.set_label("Similarity Score", fontsize=12)

plt.tight_layout()

# 🔥 **Simpan Heatmap ke file**
os.makedirs("output", exist_ok=True)  # Buat folder output jika belum ada
plt.savefig("output/heatmap_similarity.png", dpi=300, bbox_inches="tight")

# 🔥 **Tampilkan Heatmap**
plt.show()