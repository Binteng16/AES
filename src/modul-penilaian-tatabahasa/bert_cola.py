import torch
import spacy
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# === 1. Gunakan GPU jika tersedia ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

# === 2. Download & Load CoLA Dataset ===
def load_data():
    return load_dataset("glue", "cola")

dataset = load_data()

# === 3. Tokenisasi & Labeling ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(preprocess_data, batched=True, batch_size=100)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === 4. Membangun Model BERT ===
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    return model

model = load_model()

# === 5. Training Model ===
training_args = TrainingArguments(
    output_dir="./bert_cola_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

print("Melatih model... (Silakan tunggu)")
trainer.train()
print("Model telah dilatih!")

# === 6. Menyimpan Model dalam Format Universal ===
save_path = "./bert_cola_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Simpan model dalam format `pytorch_model.bin`
model_path = f"{save_path}/pytorch_model.bin"
torch.save(model.state_dict(), model_path)

print(f"Model telah disimpan di: {save_path}")
print(f"Model universal tersedia di: {model_path}")
