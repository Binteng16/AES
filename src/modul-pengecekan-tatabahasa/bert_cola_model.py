import torch
import spacy
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# === 1. Gunakan GPU jika tersedia ===
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Menggunakan perangkat: {device}")

# === 2. Download & Load CoLA Dataset ===
def load_data():
    return load_dataset("glue", "cola")

def preprocess_data(example, tokenizer):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=128)

# === 3. Tokenisasi & Labeling ===
def prepare_dataset():
    dataset = load_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
    dataset = dataset.train_test_split(test_size=0.1)  # Pembagian eksplisit training dan validasi
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset, tokenizer

dataset, tokenizer = prepare_dataset()

# === 4. Membangun Model BERT ===
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)
    return model

model = load_model()

# === 5. Training Model ===
def train_model(model, dataset):
    training_args = TrainingArguments(
        output_dir="./bert_cola_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
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
        eval_dataset=dataset["test"],
    )

    print("Melatih model... (Silakan tunggu)")
    trainer.train()
    print("Model telah dilatih!")
    return trainer

trainer = train_model(model, dataset)

# === 6. Menyimpan Model dalam Format Universal ===
def save_model(model, tokenizer, save_path="./bert_cola_model"):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")
    torch.onnx.export(model, (torch.randint(0, 30522, (1, 128)), torch.ones(1, 128)), f"{save_path}/model.onnx", input_names=["input_ids", "attention_mask"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "logits": {0: "batch_size"}})
    print(f"Model telah disimpan di: {save_path}")
    print(f"Model universal tersedia di: {save_path}/model.onnx")

save_model(model, tokenizer)