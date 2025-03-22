import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load dataset yang sudah dipreproses sebelumnya
data_preprocessing = pd.read_csv('training_data.csv')

# Inisialisasi tokenizer dan model BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Memindahkan model BERT ke perangkat (CPU atau GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Ekstraksi teks esai dari dataset
essay_texts = data_preprocessing['essay'].tolist()

# Tokenisasi teks esai dengan padding dan truncation
inputs = tokenizer(essay_texts, return_tensors='pt', padding=True, max_length=512, truncation=True)

# Ekstraksi token IDs dan attention mask
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# Mendefinisikan model untuk penilaian struktur esai
class StructureScoreModel(nn.Module):
    def __init__(self, hidden_size=768):
        super(StructureScoreModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=400, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=800, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(128 * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.dense(x)
        x = self.relu(x)
        return x

# Inisialisasi model dan pindahkan ke perangkat yang sesuai
model_structure = StructureScoreModel(hidden_size=768).to(device)

# Menyiapkan skor struktur untuk pelatihan
structure_scores = torch.tensor(data_preprocessing['skor_struktur_normalized'].values, dtype=torch.float).to(device)

# Membuat TensorDataset dan DataLoader untuk batching
train_dataset = TensorDataset(input_ids, attention_mask, structure_scores)
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Mendefinisikan fungsi loss dan optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_structure.parameters(), lr=1e-5)

# Jumlah epoch untuk pelatihan
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model_structure.train()
    optimizer.zero_grad()

    for batch in train_dataloader:
        input_ids_batch, attention_mask_batch, structure_scores_batch = batch
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        structure_scores_batch = structure_scores_batch.to(device)

        with torch.no_grad():
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
        bert_embeddings = outputs.last_hidden_state
        predictions = model_structure(bert_embeddings)

        loss = criterion(predictions.squeeze(), structure_scores_batch.squeeze())
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete!")