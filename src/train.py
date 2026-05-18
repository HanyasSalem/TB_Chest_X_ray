print ("Starting training script...")
import torch
from dataset import load_tb_data
dataset_root = r"D:\TB_project\TB_radiology_task\TB_Chest_X_ray\data"
print ("Dataset root path:", dataset_root)
def main():
    print("Starting data loading process...")

    # dataset_root = dataset_root_

    train_loader, val_loader, train_dataset, val_dataset, full_data = load_tb_data(
        dataset_root_path=dataset_root,
        batch_size=32,
        train_split_ratio=0.8
    )

    for images, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels)
        break

    print("Data loaded successfully")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

if __name__ == "__main__":
    main()
