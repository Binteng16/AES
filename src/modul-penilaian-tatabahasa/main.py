import os
import sys
import datetime
import torch
import spacy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from textblob import TextBlob

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

# === 3. Ekstraksi Kalimat dan Klausa ===
nlp = spacy.load("en_core_web_sm")

def extract_sentences(text):
    """ Memisahkan teks menjadi daftar kalimat menggunakan SpaCy. """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_clauses(sentence):
    """ Memisahkan kalimat menjadi klausa berdasarkan konjungsi dan tanda baca. """
    doc = nlp(sentence)
    clauses = []
    clause = []
    for token in doc:
        clause.append(token.text)
        if token.dep_ in ("cc", "mark", "punct") and token.text in ",.;":
            clauses.append(" ".join(clause).strip())
            clause = []
    if clause:
        clauses.append(" ".join(clause).strip())
    return clauses

# === 4. Prediksi Tata Bahasa ===
def predict_grammar(sentences):
    """ Memprediksi kesalahan tata bahasa pada setiap kalimat atau klausa. """
    if not sentences:
        return []

    encodings = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encodings = {key: val.to(device) for key, val in encodings.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.tolist()

# === 5. Setup Folder & Logging ===
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_folder = "data/result/penilaian_tata_bahasa"

os.makedirs(log_folder, exist_ok=True)

log_file_txt = os.path.join(log_folder, f"grammar_check_log_{timestamp}.txt")
log_file_csv = os.path.join(log_folder, f"grammar_check_history_{timestamp}.csv")
log_file_errors = os.path.join(log_folder, f"word_errors_log_{timestamp}.txt")

# === 6. Proses Setiap Esai ===
results_list = []
word_errors_list = []

with open(log_file_txt, "w", encoding="utf-8") as log_txt, open(log_file_errors, "w", encoding="utf-8") as log_err:
    log_txt.write(f"==== Penilaian Tata Bahasa - {timestamp} ====\n\n")
    log_err.write(f"==== Log Kesalahan Kata - {timestamp} ====\n\n")

    for idx, essay in enumerate(essays):
        print(f"\nMenilai tata bahasa pada esai {idx+1}...\n")
        sentences = extract_sentences(essay)
        sentence_results = predict_grammar(sentences)

        total_sentences = len(sentences)
        num_errors = sum(sentence_results)
        total_words = sum(len(sent.split()) for sent in sentences)
        incorrect_words = 0

        total_clauses = 0
        incorrect_clauses = 0

        incorrect_words_textblob = []
        for sent, res in zip(sentences, sentence_results):
            clauses = extract_clauses(sent)
            total_clauses += len(clauses)
            clause_results = predict_grammar(clauses)
            incorrect_clauses += sum(clause_results)

            log_txt.write(f"\nKalimat: {sent}\n")
            log_txt.write(f"Klausa:\n")
            for clause, clause_res in zip(clauses, clause_results):
                clause_status = "Benar" if clause_res == 0 else "Salah"
                log_txt.write(f"  - {clause} [{clause_status}]\n")

            if res == 1:
                blob = TextBlob(sent)
                corrected = blob.correct()
                errors = [(orig, corr) for orig, corr in zip(blob.words, corrected.words) if orig != corr]
                incorrect_words_textblob.extend(errors)

        incorrect_words = len(incorrect_words_textblob)
        WW_ratio = round((incorrect_words / total_words) * 100, 2) if total_words > 0 else 0
        CW_ratio = round((incorrect_clauses / total_clauses), 2) if total_clauses > 0 else 0

        # Menyimpan kesalahan kata ke dalam log
        if incorrect_words_textblob:
            log_err.write(f"\n--- Kesalahan Kata dalam Esai {idx+1} ---\n")
            log_err.write(f"Esai: {essay[:100]}...\n")  # Hanya tampilkan 100 karakter pertama agar tidak terlalu panjang
            log_err.write("----------------------------------------------------\n")

            for orig, corr in incorrect_words_textblob:
                log_err.write(f"Kata salah: {orig} -> Koreksi: {corr}\n")

            # Menyimpan kalimat asli & setelah koreksi
            corrected_essay = TextBlob(essay).correct()
            log_err.write("\nKalimat Asli:\n")
            log_err.write(essay + "\n")
            log_err.write("\nKalimat Setelah Koreksi:\n")
            log_err.write(str(corrected_essay) + "\n")
            log_err.write("----------------------------------------------------\n\n")

        log_txt.write("\n--- Fitur Tata Bahasa ---\n")
        log_txt.write(f"Fitur Kalimat:\n")
        log_txt.write(f"   - Total Kalimat (ST)      : {total_sentences}\n")
        log_txt.write(f"   - Kalimat Salah (SW)      : {num_errors}\n")
        log_txt.write(f"   - Rasio Kalimat Salah     : {round(num_errors / total_sentences, 2) if total_sentences > 0 else 0}\n")
        log_txt.write(f"Fitur Kata:\n")
        log_txt.write(f"   - Total Kata (WT)         : {total_words}\n")
        log_txt.write(f"   - Kata Salah (WW)         : {incorrect_words}\n")
        log_txt.write(f"   - Rasio Kata Salah        : {WW_ratio}\n")
        log_txt.write(f"Fitur Klausa:\n")
        log_txt.write(f"   - Total Klausa (CT)       : {total_clauses}\n")
        log_txt.write(f"   - Klausa Salah (CW)       : {incorrect_clauses}\n")
        log_txt.write(f"   - Rasio Klausa Salah      : {CW_ratio}\n")
        log_txt.write(f"-------------------------\n\n")

        log_err.write("\n--- Kesalahan Kata Berdasarkan TextBlob ---\n")
        for orig, corr in incorrect_words_textblob:
            log_err.write(f"{orig} -> {corr}\n")

        results_list.append({
            "timestamp": timestamp,
            "essay": essay,
            "errors": num_errors,
            "total_sentences": total_sentences,
            "total_clauses": total_clauses,
            "correct_clauses": total_clauses - incorrect_clauses,
            "incorrect_clauses": incorrect_clauses,
            "ST_ratio": round(num_errors / total_sentences, 2) if total_sentences > 0 else 0,
            "WW_ratio": WW_ratio,
            "CW_ratio": CW_ratio
        })

# Simpan hasil ke dalam CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(log_file_csv, index=False)

print(f"\nHasil penilaian telah disimpan ke: {log_file_csv}")
print(f"Log history disimpan di: {log_file_txt}")