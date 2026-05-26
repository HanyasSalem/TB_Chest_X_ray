import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# importing my model and dataset
from model import TBCNN
from dataset import load_tb_data

# parsing dataset path
import argparse 

dataset_root = r"D:\TB_project\TB_radiology_task\TB_Chest_X_ray\data"
print("Dataset root path:", dataset_root)

def get_args():
    parser = argparse.ArgumentParser(description="Train TBCNN on TB Chest X-ray dataset")
    parser.add_argument("--dataset_root", type=str, default=dataset_root, help="Path to the dataset root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--checkpoint_path", type=str, default=r"D:\TB_project\TB_radiology_task\TB_Chest_X_ray\Checkpoints", help="Directory to save model checkpoints")

    args = parser.parse_args()
    return args

def main():

    print("Starting data loading process...")
    args = get_args()
    train_loader, val_loader, train_dataset, val_dataset, full_data = load_tb_data(
        dataset_root_path=dataset_root,
        batch_size=args.batch_size,
        train_split_ratio=args.train_split_ratio
    )
    best_val_acc = 0.0
    checkpoint_path = args.checkpoint_path
    # quick sanity check
    for images, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels)
        break

    # -----------------------------
    # MODEL, LOSS, OPTIMIZER
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("Using device:", device)
    model = TBCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    num_epochs = args.num_epochs

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(output)
            predicted = (preds > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                output = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item()

                preds = torch.sigmoid(output)
                predicted = (preds > 0.5).float()

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        val_loss = val_loss / len(val_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch | {epoch + 1} | "
            f"Train loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_path}/best_model.pth")
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

    # Save final model after all epochs
    torch.save(model.state_dict(), f"{checkpoint_path}/final_model.pth")
    print("Final model saved successfully!")


if __name__ == "__main__":
    main()

