import os
import sys
import datetime
import torch
import spacy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

# === 1. Load Model dan Tokenizer ===
MODEL_PATH = "./bert_cola_model"  # Pastikan model sudah tersimpan di path ini

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

try:
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    print("Model dan tokenizer berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# === 2. Load Data dari CSV ===
CSV_PATH = r"E:\Bebeb\NEW\AES\data\result\pre_processing_data.csv"

try:
    df = pd.read_csv(CSV_PATH)
    if "essay" not in df.columns:
        raise KeyError("Kolom 'essay' tidak ditemukan dalam CSV.")
    essays = df["essay"].dropna().tolist()
    print(f"{len(essays)} esai berhasil dimuat dari '{CSV_PATH}'")
except Exception as e:
    print(f"Error saat membaca CSV: {e}")
    exit()

# === 3. Ekstraksi Kalimat dari Esai (Menggunakan Spacy) ===
nlp = spacy.load("en_core_web_sm")

def extract_sentences(text):
    """ Memisahkan teks menjadi daftar kalimat menggunakan SpaCy. """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

# === 4. Prediksi Tata Bahasa ===
def predict_grammar(sentences):
    """ Memprediksi kesalahan tata bahasa pada setiap kalimat. """
    encodings = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.tolist()

# === 5. Setup Folder & Logging ===
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_folder = "data/result/pengecekan_tata_bahasa"

# Buat folder jika belum ada
os.makedirs(log_folder, exist_ok=True)

# File log dengan timestamp
log_file_txt = os.path.join(log_folder, f"grammar_check_log_{timestamp}.txt")
log_file_csv = os.path.join(log_folder, f"grammar_check_history_{timestamp}.csv")

# === 6. Proses Setiap Esai ===
results_list = []

with open(log_file_txt, "w", encoding="utf-8") as log_txt:
    log_txt.write(f"==== Pengecekan Tata Bahasa - {timestamp} ====\n")

    for idx, essay in enumerate(essays):
        print(f"\nMengecek tata bahasa pada esai {idx+1}...\n")
        sentences = extract_sentences(essay)
        results = predict_grammar(sentences)

        num_errors = sum(results)
        total_sentences = len(sentences)

        print(f"\nJumlah kesalahan tata bahasa: {num_errors} dari {total_sentences} kalimat\n")
        log_txt.write(f"\nEsai {idx+1} - Total Kalimat: {total_sentences}, Kesalahan: {num_errors}\n")

        for i, (sent, res) in enumerate(zip(sentences, results)):
            status = "Benar" if res == 0 else "Salah"
            print(f"{i+1}. {sent} - {status}")
            log_txt.write(f"{i+1}. {sent} - {status}\n")

        # Simpan hasil ke daftar untuk CSV
        results_list.append({"timestamp": timestamp, "essay": essay, "errors": num_errors, "total_sentences": total_sentences})

# Simpan log ke CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(log_file_csv, index=False)

print(f"\nHasil pengecekan telah disimpan ke: {log_file_csv}")
print(f"Log history disimpan di: {log_file_txt}")