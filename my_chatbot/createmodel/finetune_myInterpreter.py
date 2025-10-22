import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

bertModel = 'bert-base-uncased'
# bertModel = 'distilbert-base-uncased'  # alternative

n_epochs = 50

# Load the dataset untuk Pascasarjana ITS
data = {
    "text": [
        # greet (9 samples)
        "halo", "hai", "selamat pagi", "selamat siang", "hello", "hi",
        "halo semuanya", "hai admin", "selamat pagi admin",

        # goodbye (8 samples)
        "terima kasih", "sampai jumpa", "bye", "goodbye", "selesai",
        "terima kasih banyak", "sampai bertemu lagi", "bye bye",

        # thank_you (7 samples)
        "makasih", "terima kasih", "thanks", "thank you",
        "makasih banyak", "terima kasih ya", "thanks ya",

        # inform (6 samples)
        "saya ingin bertanya", "mau tanya info", "bisa bantu saya",
        "saya perlu informasi", "mau tanya dong", "tolong bantu",

        # tanya_biaya_kuliah (12 samples)
        "berapa biaya magister", "biaya kuliah s3", "biaya doktor",
        "berapa biaya <jenjang>", "biaya kuliah <jenjang>", "biaya <jenjang> <prodi>",
        "berapa biaya magister teknik informatika", "biaya kuliah s2 elektro",
        "biaya doktor statistika", "ukt pascasarjana", "biaya pendaftaran", "info biaya kuliah",

        # tanya_prodi_tersedia (9 samples)
        "apa saja prodi di ELECTICS", "daftar prodi fakultas teknologi kelautan",
        "prodi apa saja di <fakultas>", "program studi di <fakultas>",
        "ada prodi apa di INDSYS", "jurusan di SCIENTICS", "prodi CIVPLAN",
        "program studi di teknologi elektro dan informatika cerdas", "daftar prodi",

        # cari_fakultas_dari_prodi (9 samples)
        "teknik informatika masuk fakultas apa", "teknik elektro di fakultas mana",
        "<prodi> masuk fakultas apa", "<prodi> di fakultas mana",
        "fakultas apa untuk statistika", "manajemen teknologi fakultas apa",
        "arsitektur masuk ke fakultas apa", "prodi teknik sipil fakultas mana", "fakultas untuk <prodi>",

        # beasiswa_info (8 samples)
        "apakah ada beasiswa pascasarjana", "saya ingin tahu tentang beasiswa",
        "info beasiswa", "beasiswa apa saja yang ada", "beasiswa s2",
        "beasiswa doktor", "cara daftar beasiswa", "syarat beasiswa",

        # cara_mendaftar (6 samples)
        "bagaimana cara mendaftar", "prosedur pendaftaran", "cara daftar",
        "gimana cara daftar pascasarjana", "proses pendaftaran", "daftar online",

        # jadwal_pendaftaran (5 samples)
        "kapan jadwal pendaftaran dibuka", "periode pendaftaran pascasarjana",
        "waktu pendaftaran", "jadwal buka pendaftaran", "periode daftar",

        # jadwal_ujian (6 samples)
        "tanggal ujian seleksi pascasarjana", "kapan ujian", "jadwal ujian",
        "waktu ujian masuk", "tanggal tes", "jadwal seleksi",

        # syarat_pendaftaran (5 samples)
        "apa saja syarat pendaftaran", "persyaratan masuk pascasarjana",
        "syarat masuk", "persyaratan daftar", "dokumen yang diperlukan",

        # tanya_jalur_pendaftaran (6 samples)
        "jalur pendaftaran pascasarjana", "ada jalur apa saja", "jalur masuk",
        "jalur penerimaan", "cara masuk pascasarjana", "jalur seleksi",

        # nlu_fallback (8 samples)
        "tidak jelas", "apa maksudnya", "saya tidak mengerti", "bingung",
        "kurang paham", "tidak faham", "apa itu", "maksudnya gimana"
    ],

    "label": [
        # greet (9 samples)
        "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet",

        # goodbye (8 samples)
        "goodbye", "goodbye", "goodbye", "goodbye", "goodbye",
        "goodbye", "goodbye", "goodbye",

        # thank_you (7 samples)
        "thank_you", "thank_you", "thank_you", "thank_you",
        "thank_you", "thank_you", "thank_you",

        # inform (6 samples)
        "inform", "inform", "inform", "inform", "inform", "inform",

        # tanya_biaya_kuliah (12 samples)
        "tanya_biaya_kuliah", "tanya_biaya_kuliah", "tanya_biaya_kuliah",
        "tanya_biaya_kuliah", "tanya_biaya_kuliah", "tanya_biaya_kuliah",
        "tanya_biaya_kuliah", "tanya_biaya_kuliah", "tanya_biaya_kuliah",
        "tanya_biaya_kuliah", "tanya_biaya_kuliah", "tanya_biaya_kuliah",

        # tanya_prodi_tersedia (9 samples)
        "tanya_prodi_tersedia", "tanya_prodi_tersedia", "tanya_prodi_tersedia",
        "tanya_prodi_tersedia", "tanya_prodi_tersedia", "tanya_prodi_tersedia",
        "tanya_prodi_tersedia", "tanya_prodi_tersedia", "tanya_prodi_tersedia",

        # cari_fakultas_dari_prodi (9 samples)
        "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi",
        "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi",
        "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi", "cari_fakultas_dari_prodi",

        # beasiswa_info (8 samples)
        "beasiswa_info", "beasiswa_info", "beasiswa_info", "beasiswa_info",
        "beasiswa_info", "beasiswa_info", "beasiswa_info", "beasiswa_info",

        # cara_mendaftar (6 samples)
        "cara_mendaftar", "cara_mendaftar", "cara_mendaftar",
        "cara_mendaftar", "cara_mendaftar", "cara_mendaftar",

        # jadwal_pendaftaran (5 samples)
        "jadwal_pendaftaran", "jadwal_pendaftaran", "jadwal_pendaftaran",
        "jadwal_pendaftaran", "jadwal_pendaftaran",

        # jadwal_ujian (6 samples)
        "jadwal_ujian", "jadwal_ujian", "jadwal_ujian",
        "jadwal_ujian", "jadwal_ujian", "jadwal_ujian",

        # syarat_pendaftaran (5 samples)
        "syarat_pendaftaran", "syarat_pendaftaran", "syarat_pendaftaran",
        "syarat_pendaftaran", "syarat_pendaftaran",

        # tanya_jalur_pendaftaran (6 samples)
        "tanya_jalur_pendaftaran", "tanya_jalur_pendaftaran", "tanya_jalur_pendaftaran",
        "tanya_jalur_pendaftaran", "tanya_jalur_pendaftaran", "tanya_jalur_pendaftaran",

        # nlu_fallback (8 samples)
        "nlu_fallback", "nlu_fallback", "nlu_fallback", "nlu_fallback",
        "nlu_fallback", "nlu_fallback", "nlu_fallback", "nlu_fallback"
    ]
}


df = pd.DataFrame(data)

texts = list(df["text"])
labels = df["label"].values

categories = np.unique(labels)
sample_size = len(texts)
num_class = len(categories)

print(f"Total samples: {sample_size}")
print(f"Number of classes: {num_class}")
print(f"Classes: {categories}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.3, random_state=42)

# Tokenizer and Dataset Class
tokenizer = BertTokenizer.from_pretrained(bertModel)

class PascaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

print("Prepare the datasets")

# Prepare the datasets
train_dataset = PascaDataset(X_train, y_train, tokenizer)
test_dataset = PascaDataset(X_test, y_test, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

print("Load the BERT model")

# Load BERT model
model = BertForSequenceClassification.from_pretrained(bertModel, num_labels=num_class)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Device yang dipakai: ", device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# Save the model
def save_model(model, tokenizer, model_name="pasca_intent_model"):
    # ======  Save Model and Tokenizer ======
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    import pickle
    with open(model_name+"/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to {model_name}/")
    return

# Training loop
def train_model(model, train_loader, loss_fn, optimizer, device, epochs=n_epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / len(train_loader.dataset)
        train_losses.append(avg_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return train_losses

# Evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    return accuracy

# Train the model
print("Starting training...")
train_losses = train_model(model, train_loader, loss_fn, optimizer, device)

print("Saving model...")
save_model(model, tokenizer)

print("Evaluating model...")
accuracy = evaluate_model(model, test_loader, device)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training loss')
plt.title('Training Loss over Epochs - Pascasarjana ITS Intent Classifier')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")