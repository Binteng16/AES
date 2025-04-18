import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from fitur_tata_bahasa import extract_grammar_features
import numpy as np

# Load data
df = pd.read_csv("E:/Bebeb/NEW/AES/data/pre-processing-data/training/training_data.csv")

# Fitur numerik yang akan digunakan
fitur_numerik = [
    "STg", "SWg", "SWg_new", "WTg", "Ws", "CTg", "CWg", "CWg_new",
    "WTg/STg", "WTg/CTg", "SWg/STg", "CWg/CTg", "Ws/WTg"
]

fitur_semua = []
target_semua = []

print("Mengekstrak fitur grammar dari semua esai...")
for idx, row in df.iterrows():
    fitur = extract_grammar_features(row["essay"])["summary"]
    vektor_fitur = [fitur[k] for k in fitur_numerik]
    skor = row["skor_tata_bahasa_normalized"]
    
    fitur_semua.append(vektor_fitur)
    target_semua.append(skor)

# Split data
X_train, X_val, y_train, y_val = train_test_split(fitur_semua, target_semua, test_size=0.2, random_state=42)

# Scaling fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Cek NaN (opsional, keamanan)
if np.any(pd.isnull(X_train)) or np.any(pd.isnull(X_val)):
    print("Data mengandung nilai kosong, harap periksa dan bersihkan!")
    exit()

# Train logistic regression
print("Melatih model Logistic Regression...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Simpan model dan scaler
joblib.dump(model, "modul_penilaian_tata_bahasa/model_logistic_regression.pkl")
joblib.dump(scaler, "modul_penilaian_tata_bahasa/scaler.pkl")
print("Model dan scaler disimpan sebagai model_logistic_regression.pkl dan scaler.pkl")