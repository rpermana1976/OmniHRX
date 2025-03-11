import os
import pdfplumber
import fitz  # PyMuPDF
import docx
import chardet

def extract_text_from_pdf(pdf_path):
    """Ekstrak teks dari PDF dengan pdfplumber & fallback ke PyMuPDF."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"⚠️ pdfplumber gagal di {pdf_path}, mencoba PyMuPDF... ({e})")
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text("text") for page in doc])
        except Exception as e:
            print(f"❌ Gagal ekstrak PDF {pdf_path}: {e}")

    return text.strip()

def extract_text_from_docx(docx_path):
    """Ekstrak teks dari file DOCX dengan python-docx."""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"❌ Gagal ekstrak DOCX {docx_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Ekstrak teks dari file TXT dengan deteksi encoding otomatis."""
    try:
        with open(txt_path, "rb") as f:
            raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding'] or "utf-8"

        with open(txt_path, "r", encoding=encoding) as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Gagal ekstrak TXT {txt_path}: {e}")
        return ""

def extract_text_from_file(file_path):
    """Deteksi format file dan panggil fungsi ekstraksi yang sesuai."""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        print(f"⚠️ Format tidak dikenali: {file_path}")
        return ""

def process_folder(input_folder, output_folder):
    """Memproses semua file dalam folder dan menyimpan hasil ekstraksi teks."""
    if not os.path.exists(input_folder):
        print(f"⚠️ Folder tidak ditemukan: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename + ".txt")

        if os.path.isfile(input_file_path):
            print(f"📄 Mengekstrak: {filename}")
            text = extract_text_from_file(input_file_path)

            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Teks diekstrak ke: {output_file_path}")

if __name__ == "__main__":
    print("🚀 Memulai ekstraksi teks...")
    process_folder("data/raw_data/resumes", "data/processed_data/resumes")
    process_folder("data/raw_data/jobs", "data/processed_data/jobs")
    print("🎉 Ekstraksi teks selesai!")
