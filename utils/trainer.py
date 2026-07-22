import copy
import json
import os

import torch    
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    model_save_path,
):

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    
    patience = 5
    counter = 0

    os.makedirs("outputs/history", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 50)

        # -----------------------------
        # Training
        # -----------------------------
        model.train()

        running_loss = 0.0
        train_preds = []
        train_labels = []

        for batch_idx, (images, labels) in enumerate(train_loader):

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"Training Batch {batch_idx+1}/{len(train_loader)}")

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(val_loader):

                if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                    print(f"Validation Batch {batch_idx+1}/{len(val_loader)}")

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        precision = precision_score(
            val_labels,
            val_preds,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            val_labels,
            val_preds,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            val_labels,
            val_preds,
            average="weighted",
            zero_division=0
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print("\nEpoch Summary")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        if val_acc > best_accuracy:

            best_accuracy = val_acc
            best_model = copy.deepcopy(model.state_dict())

            torch.save(best_model, model_save_path)

            counter = 0

            print("✅ Best Model Saved")

        else:

            counter += 1
            print(f"No Improvement ({counter}/{patience})")
            
        if counter >= patience:
            print("\n🛑 Early Stopping Triggered!")
            break


    with open(
        "outputs/history/training_history.json",
        "w"
    ) as f:

        json.dump(history, f, indent=4)

    model.load_state_dict(best_model)

    print("\nTraining Completed.")
    print(f"Best Validation Accuracy : {best_accuracy:.4f}")
    
    history["best_validation_accuracy"] = best_accuracy

    return model, history