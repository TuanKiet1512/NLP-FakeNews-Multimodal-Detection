
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. LOAD & CÂN BẰNG DỮ LIỆU
# ==========================================
data_path = '/content/drive/MyDrive/dataset/multimodal_train.tsv'
data_full = pd.read_csv(data_path, sep='\t')
data_full = data_full.dropna(subset=['clean_title', '2_way_label'])

# Lấy đều mỗi bên 8,000 mẫu để tạo bộ dữ liệu 16,000 dòng (Tránh thiên kiến)
data_fake = data_full[data_full['2_way_label'] == 0].sample(8000, random_state=42)
data_real = data_full[data_full['2_way_label'] == 1].sample(8000, random_state=42)

data = pd.concat([data_fake, data_real]).sample(frac=1, random_state=42).reset_index(drop=True)
y = data['2_way_label'].values

print(f"✅ Đã nạp & cân bằng: {len(data)} mẫu (8000 Real / 8000 Fake)")

# ==========================================
# 2. CẤU HÌNH & SIÊU THAM SỐ
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 16 # Tăng lên 16 nếu GPU ổn định
EPOCHS = 4      # Giảm epoch để tránh overfitting sớm
LEARNING_RATE = 1e-5 # Hạ thấp LR để học sâu hơn

# ==========================================
# 3. DATASET & MODEL
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, transform):
        self.data = dataframe
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.missing_images = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_title'])
        label = int(row['2_way_label'])
        img_id = row['id']

        encoding = self.tokenizer(
            text, truncation=True, add_special_tokens=True,
            max_length=MAX_LEN, padding='max_length',
            return_attention_mask=True, return_tensors='pt'
        )

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            # Nếu mất ảnh, trả về Tensor 0 (ảnh đen)
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalClassifier(nn.Module):
    def __init__(self, n_classes):
        super(MultimodalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # Fusion Layer
        self.fusion = nn.Linear(768 + 2048, 512)
        self.dropout = nn.Dropout(0.7) # Tăng Dropout mạnh để chống học vẹt 100%
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, input_ids, attention_mask, image):
        # Text features từ BERT
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        # Image features từ ResNet
        image_features = self.resnet(image)

        # Kết hợp (Concatenate)
        combined = torch.cat((text_features, image_features), dim=1)
        x = torch.relu(self.fusion(combined))
        x = self.dropout(x)
        return self.classifier(x)

# ==========================================
# 4. CHUẨN BỊ TRAINING
# ==========================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_df, val_df = train_test_split(data, test_size=0.2, random_state=42, stratify=y)
img_dir = "/content/drive/MyDrive/fakeddit_images/"

train_loader = DataLoader(MultimodalDataset(train_df, img_dir, tokenizer, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultimodalDataset(val_df, img_dir, tokenizer, transform), batch_size=BATCH_SIZE)

model = MultimodalClassifier(n_classes=2).to(DEVICE)

# Thêm weight_decay để kiểm soát độ lớn của tham số, giảm overfitting
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. VÒNG LẶP HUẤN LUYỆN
# ==========================================
def train_model(model, train_loader, val_loader, epochs=4):
    best_acc = 0
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Phase: Train
        model.train()
        losses, correct = [], 0
        for d in train_loader:
            input_ids = d["input_ids"].to(DEVICE)
            masks = d["attention_mask"].to(DEVICE)
            images = d["image"].to(DEVICE)
            labels = d["label"].to(DEVICE)

            outputs = model(input_ids, masks, images)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_acc = correct.double() / len(train_loader.dataset)

        # Phase: Eval
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for d in val_loader:
                outputs = model(d["input_ids"].to(DEVICE), d["attention_mask"].to(DEVICE), d["image"].to(DEVICE))
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(d["label"].cpu().tolist())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Train Loss: {np.mean(losses):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            torch.save(model.state_dict(), 'best_multimodal_model.bin')
            best_acc = val_acc
            print("⭐️ Đã lưu mô hình tốt nhất!")

# Thực thi
train_model(model, train_loader, val_loader, epochs=EPOCHS)

# ==========================================
# 6. BÁO CÁO CUỐI CÙNG (SỬA LẠI LABEL)
# ==========================================
model.load_state_dict(torch.load('best_multimodal_model.bin'))
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for d in val_loader:
        outputs = model(d["input_ids"].to(DEVICE), d["attention_mask"].to(DEVICE), d["image"].to(DEVICE))
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(d["label"].cpu().tolist())

print("\n===== KẾT QUẢ PHÂN TÍCH CHI TIẾT =====")
# Sửa thứ tự: 0 là Fake, 1 là Real
print(classification_report(all_true, all_preds, target_names=['Fake', 'Real']))