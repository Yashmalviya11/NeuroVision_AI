from utils.dataset import get_dataloaders
from utils.model_loader import load_model
from utils.trainer import train_model

import torch
import torch.nn as nn
import torch.optim as optim

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

IMAGE_SIZE = 224
BATCH_SIZE = 4    
EPOCHS = 30
LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = "models/best_model.pth"


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("NeuroVision AI - Vision Transformer Training")
    print("=" * 60)
    print(f"Using Device : {device}")
    print("=" * 60)

    # -----------------------------
    # Load Dataset
    # -----------------------------
    train_loader, test_loader, class_names = get_dataloaders(
        TRAIN_DIR,
        TEST_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        num_workers=0
    )

    print("\nDetected Classes:")
    print(class_names)

    print(f"\nTrain Images  : {len(train_loader.dataset)}")
    print(f"Test Images   : {len(test_loader.dataset)}")
    print(f"Train Batches : {len(train_loader)}")
    print(f"Test Batches  : {len(test_loader)}")

    # -----------------------------
    # Test DataLoader
    # -----------------------------
    print("\nTesting DataLoader...")

    images, labels = next(iter(train_loader))

    print("DataLoader Working")
    print("Image Shape :", images.shape)
    print("Label Shape :", labels.shape)

    # -----------------------------
    # Load Model
    # -----------------------------
    print("\nLoading Vision Transformer...")

    model = load_model(len(class_names))
    model = model.to(device)

    print("Model Loaded Successfully!")

    print(model.head)

    # -----------------------------
    # Test GPU Forward Pass
    # -----------------------------
    print("\nTesting GPU Forward Pass...")

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print("GPU Forward Pass Successful")
    print("Output Shape :", outputs.shape)
    
    criterion = nn.CrossEntropyLoss()
    
    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------
    # Scheduler
    # -----------------------------
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-4
    )

    print("\nOptimizer : AdamW")
    print("Loss      : CrossEntropyLoss")
    print("Scheduler : ReduceLROnPlateau")

    print("\nEverything Loaded Successfully.")
    print("=" * 60)

    # -----------------------------
    # Start Training
    # -----------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=EPOCHS,
        model_save_path=MODEL_SAVE_PATH
    )


if __name__ == "__main__":
    main()