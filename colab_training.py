# NOTE: This code is meant to be run in Google Colab or a Jupyter Notebook environment
# since it uses notebook "!" commands for downloading and installing libraries.

import os
import json
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models

# =======================================================
# 1. SETUP & DOWNLOADING DATA (Run in Google Colab)
# =======================================================
# !pip install torch torchvision timm scikit-learn pandas matplotlib -q
# !wget -q --show-progress "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip" -O isic2019/images.zip
# !wget -q "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv" -O isic2019/labels.csv
# !wget -q "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv" -O isic2019/metadata.csv
# !unzip -q isic2019/images.zip -d isic2019/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# =======================================================
# 2. DATA PREPARATION & FOLDER STRUCTURE
# =======================================================
df = pd.read_csv('isic2019/labels.csv')

# ISIC 2019 has 8 classes
CLASS_NAMES = {
    'MEL':   'Melanoma',
    'NV':    'Melanocytic Nevi',
    'BCC':   'Basal Cell Carcinoma',
    'AK':    'Actinic Keratosis',
    'BKL':   'Benign Keratosis',
    'DF':    'Dermatofibroma',
    'VASC':  'Vascular Lesion',
    'SCC':   'Squamous Cell Carcinoma'
}

# Convert one-hot to single label column
label_cols = list(CLASS_NAMES.keys())
df['label'] = df[label_cols].idxmax(axis=1)

# Create folder structure for PyTorch ImageFolder
base = Path('dataset')
for split in ['train', 'val']:
    for cls in CLASS_NAMES:
        (base / split / cls).mkdir(parents=True, exist_ok=True)

img_dir = Path('isic2019/ISIC_2019_Training_Input')
img_map = {f.stem: f for f in img_dir.glob('*.jpg')}

# 80/20 train/validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print("Organizing files...")
for row in train_df.itertuples():
    src = img_map.get(row.image)
    if src: shutil.copy(src, base / 'train' / row.label / f'{row.image}.jpg')

for row in val_df.itertuples():
    src = img_map.get(row.image)
    if src: shutil.copy(src, base / 'val' / row.label / f'{row.image}.jpg')


# =======================================================
# 3. DATA LOADERS & DATA AUGMENTATION
# =======================================================
IMG_SIZE  = 224
BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder('dataset/train', transform=train_transforms)
val_ds   = datasets.ImageFolder('dataset/val',   transform=val_transforms)

# Handle ISIC's massive class imbalance
class_counts    = np.array([len(list(Path(f'dataset/train/{c}').glob('*.jpg'))) for c in train_ds.classes])
sample_weights  = (1.0 / class_counts)[train_ds.targets]
sampler         = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# =======================================================
# 4. BUILDING THE RESNET-50 MODEL
# =======================================================
NUM_CLASSES = len(train_ds.classes)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze early layers
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# Replace the head for 8 classes
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(device)


# =======================================================
# 5. TRAINING LOOP
# =======================================================
EPOCHS = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = torch.max(model(imgs), 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = correct / total
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1:02d} | loss={running_loss/len(train_loader):.4f} | val_acc={val_acc:.4f} [SAVED]")
    else:
        print(f"Epoch {epoch+1:02d} | loss={running_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

# =======================================================
# 6. EXPORTING CLASSES & DOWNLOADING
# =======================================================
idx_to_class = {v: k  for k, v in train_ds.class_to_idx.items()}
idx_to_name  = {v: CLASS_NAMES[k] for k, v in train_ds.class_to_idx.items()}

with open('class_index.json', 'w') as f:
    json.dump({
        'idx_to_class': idx_to_class,
        'idx_to_name':  idx_to_name,
        'class_names':  CLASS_NAMES,
        'num_classes':  NUM_CLASSES
    }, f, indent=2)

# Un-comment in Google Colab to trigger download
# from google.colab import files
# files.download('best_model.pth')
# files.download('class_index.json')
