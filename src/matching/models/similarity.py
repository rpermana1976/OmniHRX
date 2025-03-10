import os
import sys
import numpy as np
import pickle
import torch
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

# Pastikan Python mengenali folder src
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
        return {}  # 🔥 Jika file tidak ada, return cache kosong
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):  # 🔥 Tangani error cache corrupt
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

# 🔥 **Fungsi utama untuk mencocokkan CV dengan job descriptions**
def match_resume_to_jobs(resume_text, job_texts):
    scores = []
    for job_text in job_texts:
        w2v_score = np.dot(get_sentence_vector(resume_text, word2vec_model, "cache/w2v.pkl"),
                           get_sentence_vector(job_text, word2vec_model, "cache/w2v.pkl"))
        ft_score = np.dot(get_sentence_vector(resume_text, fasttext_model, "cache/ft.pkl"),
                          get_sentence_vector(job_text, fasttext_model, "cache/ft.pkl"))
        sbert_score = sbert_similarity(resume_text, job_text)

        # 🔥 **Penyesuaian bobot untuk similarity yang lebih akurat**
        final_score = (0.10 * w2v_score) + (0.20 * ft_score) + (0.70 * sbert_score)
        scores.append(final_score)

    return scores

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

    # 🔥 **Tampilkan hasil dalam format tabel**
    print("\n📊 **Hasil Matching CV dengan Lowongan**")
    print("-" * 70)
    print(f"{'CV':<40} | {'Lowongan':<30} | {'Similarity Score':<10}")
    print("-" * 70)

    for i, resume in enumerate(resumes):
        sorted_jobs = sorted(zip(job_files, results[i]), key=lambda x: x[1], reverse=True)
        for job, score in sorted_jobs[:5]:  # Ambil 5 job terbaik
            print(f"{resume_files[i]:<40} | {job:<30} | {score:.4f}")

    print("\n🎉 Pencocokan selesai!")