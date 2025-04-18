import pandas as pd
import joblib
from fitur_tata_bahasa import extract_grammar_features

# Load model
model = joblib.load("modul_penilaian_tata_bahasa/model_logistic_regression.pkl")

# Load data
df = pd.read_csv("E:/Bebeb/NEW/AES/data/pre-processing-data/training/training_data.csv")

# List fitur yang digunakan
fitur_numerik = [
    "STg", "SWg", "SWg_new", "WTg", "Ws", "CTg", "CWg", "CWg_new",
    "WTg/STg", "WTg/CTg", "SWg/STg", "CWg/CTg", "Ws/WTg"
]

hasil_semua = []

for idx, row in df.iterrows():
    essay_id = row["essay_id"]
    skor_asli = row["skor_tata_bahasa_normalized"]
    hasil = extract_grammar_features(row["essay"])
    
    fitur = hasil["summary"]
    fitur_urutan = [fitur[k] for k in fitur_numerik]
    skor_prediksi = model.predict([fitur_urutan])[0]
    
    hasil_semua.append({
        "essay_id": essay_id,
        "skor_grammar_asli": skor_asli,
        "skor_grammar_prediksi": round(skor_prediksi, 2)
    })

hasil_df = pd.DataFrame(hasil_semua)
hasil_df.to_csv("modul_penilaian_tata_bahasa/hasil_prediksi_grammar.csv", index=False)

print("Hasil prediksi grammar disimpan di hasil_prediksi_grammar.csv")
