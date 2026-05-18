import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

# Custom Dataset to apply different transforms to subsets after random_split
class CustomImageDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Subset returns (image, label) directly from the underlying dataset (ImageFolder)
        image, label = self.subset[index]
        if self.transform:
            # ImageFolder returns PIL Image, transform expects PIL Image
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

def load_tb_data(dataset_root_path, batch_size=32, train_split_ratio=0.8):
    """
    Loads and prepares the Tuberculosis Chest X-Ray dataset.

    Args:
        dataset_root_path (str): The path to the directory containing
                                 'TB_Chest_Radiography_Database'. This is typically
                                 the output of `kagglehub.dataset_download`.
        batch_size (int): The batch size for DataLoaders.
        train_split_ratio (float): The ratio for splitting data into
                                   training and validation sets (e.g., 0.8 for 80% train).

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - train_dataset (CustomImageDataset): The transformed training dataset.
            - val_dataset (CustomImageDataset): The transformed validation dataset.
            - full_data (ImageFolder): The original full dataset (before custom transforms).
    """
    # Construct the full path to the image data directory
    data_dir = os.path.join(dataset_root_path, 'TB_Chest_Radiography_Database')
    full_data = datasets.ImageFolder(data_dir)

    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels= 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std = [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels= 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std = [0.5])
    ])

    # Split the full dataset into training and validation subsets
    train_size = int(train_split_ratio * len(full_data))
    val_size = len(full_data) - train_size
    train_subset, val_subset = random_split(full_data, [train_size, val_size])

    # Wrap subsets with CustomImageDataset to apply specific transforms
    train_dataset = CustomImageDataset(train_subset, transform=train_transform)
    val_dataset = CustomImageDataset(val_subset, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f"Successfully loaded data from: {data_dir}")
    print(f"Total images found: {len(full_data)}")
    print(f"Training set size: {len(train_dataset)} images.")
    print(f"Validation set size: {len(val_dataset)} images.")

    return train_loader, val_loader, train_dataset, val_dataset, full_data