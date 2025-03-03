import os
import fitz  # PyMuPDF untuk PDF
import docx  # python-docx untuk DOCX
import pytesseract  # OCR
import cv2  # OpenCV untuk pemrosesan gambar

# Pastikan Tesseract terpasang di sistem (ubah path sesuai instalasi)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    """Ekstrak teks dari file PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_text_from_docx(docx_path):
    """Ekstrak teks dari file DOCX."""
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_image(image_path):
    """Ekstrak teks dari gambar menggunakan OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale untuk OCR lebih akurat
    text = pytesseract.image_to_string(gray, lang="eng")  # Bisa pakai "ind" untuk Bahasa Indonesia
    return text

def extract_text(file_path):
    """Deteksi format file dan ekstrak teksnya."""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(file_path)
    else:
        print(f"❌ Format file tidak didukung: {file_path}")
        return None

def process_folder(input_folder, output_folder):
    """Memproses semua file dalam folder dan menyimpan hasil ekstraksi."""
    if not os.path.exists(input_folder):
        print(f"⚠️ Folder tidak ditemukan: {input_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename + ".txt")

        if os.path.isfile(file_path):
            print(f"📄 Memproses: {filename}")
            text = extract_text(file_path)
            if text:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✅ Teks diekstrak dan disimpan di: {output_file_path}")

if __name__ == "__main__":
    print("🚀 Memulai ekstraksi teks...")

    # Proses folder resumes dan jobs
    process_folder("data/raw_data/resumes", "data/processed_data/resumes")
    process_folder("data/raw_data/jobs", "data/processed_data/jobs")

    print("🎉 Ekstraksi selesai!")
