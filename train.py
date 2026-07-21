import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import timm

# =====================================================
# Configuration
# =====================================================

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("NeuroVision AI - Vision Transformer Training")
print("=" * 60)
print(f"Using Device : {device}")
print("=" * 60)

# =====================================================
# Image Transformations
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# Dataset
# =====================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("\nDetected Classes:")
print(train_dataset.classes)

print(f"\nTraining Images : {len(train_dataset)}")
print(f"Testing Images  : {len(test_dataset)}")

# =====================================================
# Load Vision Transformer
# =====================================================

print("\nLoading Vision Transformer...\n")

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True
)

# Replace classifier for Alzheimer's classes

model.head = nn.Linear(
    model.head.in_features,
    len(train_dataset.classes)
)

model = model.to(device)

print("Model Loaded Successfully!\n")

print(model.head)

# =====================================================
# Loss Function
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# Optimizer
# =====================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# Learning Rate Scheduler
# =====================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)

print("\nOptimizer : AdamW")
print("Loss      : CrossEntropyLoss")
print("Scheduler : ReduceLROnPlateau")

print("\nEverything Loaded Successfully.")
print("=" * 60)