# Import library yang dibutuhkan
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load dataset yang sudah dipreproses sebelumnya
data_preprocessing = pd.read_csv('pre_processing_data.csv')
"""
Baris ini membaca dataset hasil pre-processing dari file CSV bernama 'pre_processing_data.csv'
menggunakan library pandas. Dataset ini akan digunakan untuk melatih model.
"""

# Inisialisasi tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
"""
Baris ini memuat tokenizer dan model BERT yang sudah dilatih sebelumnya ('bert-base-uncased').
Tokenizer digunakan untuk memproses teks input menjadi token ID, sedangkan model BERT digunakan
untuk menghasilkan representasi fitur dari teks input.
"""

# Memindahkan model BERT ke perangkat (CPU atau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
"""
Baris ini memeriksa apakah GPU tersedia menggunakan torch.cuda.is_available().
Jika tersedia, perangkat yang digunakan adalah GPU (CUDA); jika tidak, akan menggunakan CPU.
Model BERT kemudian dipindahkan ke perangkat yang sesuai.
"""

# Ekstraksi teks esai dari dataset
essay_texts = data_preprocessing['essay'].tolist()
"""
Baris ini mengekstrak kolom 'essay' dari dataset hasil pre-processing dan mengubahnya menjadi
list Python menggunakan fungsi .tolist(). List ini berisi teks dari esai yang akan diproses.
"""

# Tokenisasi teks esai dengan padding dan truncation
inputs = tokenizer(essay_texts, return_tensors='pt', padding=True, max_length=512, truncation=True)
"""
Baris ini menggunakan tokenizer BERT untuk memproses teks esai:
- `return_tensors='pt'`: Menghasilkan tensor PyTorch.
- `padding=True`: Menambahkan padding agar semua teks memiliki panjang yang sama.
- `max_length=512`: Memotong teks yang lebih panjang dari 512 token.
- `truncation=True`: Mengaktifkan pemotongan teks yang terlalu panjang.
"""

# Ekstraksi token IDs dan attention mask
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
"""
Baris ini mengekstrak token IDs dan attention mask dari hasil tokenisasi.
- `input_ids`: Tensor yang berisi ID token untuk setiap teks.
- `attention_mask`: Tensor yang menunjukkan token mana yang relevan (1) dan mana yang berupa padding (0).
"""

# Memindahkan tokenized inputs ke perangkat yang sesuai
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
"""
Baris ini memindahkan tokenized inputs (input_ids dan attention_mask) ke perangkat yang digunakan (GPU atau CPU).
"""

# Mendefinisikan model untuk penilaian struktur esai
class StructureScoreModel(nn.Module):
    def __init__(self, hidden_size=768):
        """
        Konstruktor untuk model StructureScoreModel.
        - `hidden_size=768`: Ukuran output dari layer BERT (default 768).
        """
        super(StructureScoreModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=400, batch_first=True, bidirectional=True)
        """
        Layer LSTM pertama:
        - `input_size=hidden_size`: Inputnya adalah keluaran dari BERT (768).
        - `hidden_size=400`: Ukuran hidden state (400).
        - `batch_first=True`: Input/output memiliki dimensi (batch_size, seq_length, features).
        - `bidirectional=True`: Menggunakan LSTM dua arah, sehingga outputnya menjadi (batch_size, seq_length, 800).
        """
        
        self.lstm2 = nn.LSTM(input_size=800, hidden_size=128, batch_first=True, bidirectional=True)
        """
        Layer LSTM kedua:
        - `input_size=800`: Output dari layer LSTM pertama (400 * 2 untuk bidirectional).
        - `hidden_size=128`: Ukuran hidden state (128).
        - `batch_first=True`: Dimensi input/output tetap sama.
        - `bidirectional=True`: Outputnya menjadi (batch_size, seq_length, 256).
        """

        self.dropout = nn.Dropout(0.5)
        """
        Dropout layer dengan probabilitas 0.5 untuk mencegah overfitting.
        """

        self.dense = nn.Linear(128 * 2, 1)
        """
        Dense layer (fully connected):
        - `input_size=256` (128 * 2 untuk bidirectional).
        - `output_size=1`: Menghasilkan skor tunggal.
        """
        
        self.relu = nn.ReLU()
        """
        Fungsi aktivasi ReLU untuk memastikan keluaran positif.
        """

    def forward(self, x):
        """
        Fungsi forward untuk model StructureScoreModel.
        - `x`: Input tensor dengan dimensi (batch_size, seq_length, features).
        """
        x, _ = self.lstm1(x)
        """
        Input melalui layer LSTM pertama. Output memiliki dimensi (batch_size, seq_length, 800).
        """
        x, _ = self.lstm2(x)
        """
        Output dari LSTM pertama masuk ke LSTM kedua. Dimensi output menjadi (batch_size, seq_length, 256).
        """
        x = self.dropout(x)
        """
        Output dari LSTM kedua dikenakan dropout.
        """
        x = x[:, -1, :]
        """
        Mengambil output token terakhir dari sequence (dimensi -1).
        """
        x = self.dense(x)
        """
        Output dari token terakhir dimasukkan ke dense layer untuk menghasilkan skor tunggal.
        """
        x = self.relu(x)
        """
        Output dari dense layer dikenakan fungsi aktivasi ReLU.
        """
        return x

# Inisialisasi model dan pindahkan ke perangkat yang sesuai
model_structure = StructureScoreModel(hidden_size=768).to(device)
"""
Baris ini membuat instance dari model StructureScoreModel dan memindahkannya ke perangkat yang digunakan.
"""

# Menyiapkan skor struktur untuk pelatihan
structure_scores = torch.tensor(data_preprocessing['skor_struktur_normalized'].values, dtype=torch.float).to(device)
"""
Baris ini mengonversi skor struktur dari dataset hasil pre-processing menjadi tensor PyTorch dengan tipe data float
dan memindahkannya ke perangkat yang digunakan (GPU atau CPU).
"""

# Membagi data menjadi set pelatihan dan validasi
train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_scores, val_scores = train_test_split(
    input_ids.cpu(), attention_mask.cpu(), structure_scores.cpu(), test_size=0.2, random_state=42)
"""
Baris ini membagi data menjadi set pelatihan dan validasi dengan rasio 80:20 menggunakan fungsi train_test_split.
- `test_size=0.2`: 20% data digunakan untuk validasi.
- `random_state=42`: Seed untuk memastikan hasil pembagian konsisten.
"""

# Membuat TensorDataset dan DataLoader untuk batching
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_scores)
val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_scores)
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
"""
Baris ini membuat dataset PyTorch (TensorDataset) dari data yang telah dibagi untuk pelatihan dan validasi.
DataLoader digunakan untuk mengatur batching dan shuffle selama pelatihan.
"""

# Mendefinisikan fungsi loss dan optimizer
criterion = nn.MSELoss()
"""
Mean Squared Error (MSE) digunakan sebagai fungsi loss untuk tugas regresi ini.
"""
optimizer = torch.optim.Adam(model_structure.parameters(), lr=1e-5)
"""
Adam optimizer digunakan dengan learning rate sebesar 1e-5 untuk mengoptimalkan parameter model.
"""

""" Fungsi untuk mendiskretkan skor kontinu ke dalam kategori definisi bin """
""" Parameter `bins` menentukan jumlah kategori. """
def discretize_scores(scores, bins=10):
    """ Mendapatkan nilai minimum dan maksimum dari skor, lalu membaginya menjadi kategori. """
    min_score = np.min(scores)  # Mendapatkan nilai minimum dari skor
    max_score = np.max(scores)  # Mendapatkan nilai maksimum dari skor
    bin_edges = np.linspace(min_score, max_score, bins + 1)  # Membagi rentang skor ke dalam `bins` kategori
    discretized_scores = np.digitize(scores, bin_edges) - 1  # Menghitung kategori untuk setiap skor (0-indexed)
    return discretized_scores

""" Parameter untuk pelatihan """
""" Jumlah epoch untuk melatih model. """
num_epochs = 50

""" Training loop """
""" Proses pelatihan model selama sejumlah epoch. """
for epoch in range(num_epochs):
    """ Mengatur model ke mode pelatihan dan memulai iterasi untuk setiap epoch. """
    model_structure.train()  # Mengatur model ke mode pelatihan
    optimizer.zero_grad()  # Mengatur ulang gradien optimizer

    predicted_structure_scores_train = []  # Menyimpan skor prediksi untuk data pelatihan
    true_structure_scores_train = []  # Menyimpan skor sebenarnya untuk data pelatihan

    for batch in train_dataloader:  # Melakukan iterasi pada setiap batch data pelatihan
        """ Mengambil data input dan skor dari batch """
        input_ids_batch, attention_mask_batch, structure_scores_batch = batch

        """ Memindahkan data input dan skor ke device (GPU atau CPU). """
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        structure_scores_batch = structure_scores_batch.to(device)

        with torch.no_grad():  # Mematikan perhitungan gradien untuk model utama
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)

        bert_embeddings = outputs.last_hidden_state  # Mengambil embedding dari output model
        predictions = model_structure(bert_embeddings)  # Mendapatkan prediksi dari model struktur

        """ Menghitung loss antara prediksi dan skor sebenarnya. """
        loss = criterion(predictions.squeeze(), structure_scores_batch.squeeze())

        loss.backward()  # Melakukan backpropagation
        optimizer.step()  # Memperbarui parameter model

        """ Menyimpan prediksi dan skor sebenarnya untuk menghitung QWK. """
        predicted_structure_scores_train.extend(predictions.detach().cpu().numpy().flatten())
        true_structure_scores_train.extend(structure_scores_batch.detach().cpu().numpy())

    """ Menghitung QWK untuk data pelatihan """
    """ Skala prediksi dan skor sebenarnya ke dalam rentang 1-10 untuk menghitung QWK. """

    qwk_train_score = cohen_kappa_score(
        discretize_scores(predicted_structure_scores_train),
        discretize_scores(true_structure_scores_train),
        weights='quadratic')

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training QWK Score: {qwk_train_score:.4f}")

    """ Validation loop """
    """ Proses validasi model untuk mengevaluasi performa. """
    model_structure.eval()  # Mengatur model ke mode evaluasi
    predicted_structure_scores_val = []  # Menyimpan skor prediksi untuk data validasi
    true_structure_scores_val = []  # Menyimpan skor sebenarnya untuk data validasi
    val_loss = 0.0  # Inisialisasi nilai loss validasi

    with torch.no_grad():  # Mematikan perhitungan gradien untuk validasi
        for batch in val_dataloader:  # Iterasi pada setiap batch data validasi
            """ Mengambil data input dan skor dari batch """
            input_ids_batch, attention_mask_batch, structure_scores_batch = batch

            """ Memindahkan data input dan skor ke device (GPU atau CPU). """
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            structure_scores_batch = structure_scores_batch.to(device)

            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)  # Mendapatkan output dari model
            bert_embeddings = outputs.last_hidden_state  # Mengambil embedding dari output model

            predictions = model_structure(bert_embeddings)  # Mendapatkan prediksi dari model struktur

            """ Menghitung dan menjumlahkan loss untuk setiap batch. """
            val_loss += criterion(predictions.squeeze(), structure_scores_batch.squeeze()).item()

            """ Menyimpan prediksi dan skor sebenarnya untuk menghitung QWK. """
            predicted_structure_scores_val.extend(predictions.detach().cpu().numpy().flatten())
            true_structure_scores_val.extend(structure_scores_batch.detach().cpu().numpy())

    val_loss = val_loss / len(val_dataloader)  # Menghitung rata-rata loss validasi

    qwk_val_score = cohen_kappa_score(
        discretize_scores(predicted_structure_scores_val),
        discretize_scores(true_structure_scores_val),
        weights='quadratic')

    print(f"Validation Loss: {val_loss:.4f}, Validation QWK Score: {qwk_val_score:.4f}")

""" Menyimpan prediksi mentah dan tereskala ke dalam file CSV """
""" Menyimpan hasil prediksi ke file CSV untuk analisis lebih lanjut. """
predictions_combined_df = pd.DataFrame({
    'true_scores_raw': true_structure_scores_val,  # Skor sebenarnya (mentah)
    'predicted_scores_raw': predicted_structure_scores_val,  # Skor prediksi (mentah)
    'true_scores_scaled': true_structure_scores_val,  # Skor sebenarnya
    'predicted_scores_scaled': predicted_structure_scores_val  # Skor prediksi
})

""" Menyimpan dataframe ke file CSV """
""" Menyimpan dataframe yang telah berisi data prediksi dan skor sebenarnya ke dalam file CSV. """
predictions_combined_df.to_csv('predicted_structure_scores_combined.csv', index=False)

print("Training complete!")