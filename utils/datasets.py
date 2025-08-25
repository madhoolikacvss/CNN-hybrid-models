import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import kaggle


def ensure_dataset(dataset_id: str, local_path: str):
    """Download dataset from Kaggle if not found locally."""
    if not os.path.exists(local_path):
        print(f"Downloading {dataset_id} to {local_path}...")
        kaggle.api.dataset_download_files(dataset_id, local_path, unzip=True)
        print("Downloaded to:", local_path)
    else:
        print(f"Using existing dataset at {local_path}")
    return local_path


def get_transforms(train=True):
    """Define data augmentation and preprocessing transforms."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(20),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])


def get_newplant_loaders(root="./data", batch_size=32):
    """
    Returns train, val, test DataLoaders for the New Plant Disease Dataset Augmented.
    Dataset structure should be:
    root/New plant disease dataset augmented/train/
    root/New plant disease dataset augmented/valid/
    root/test/
    """
    # Ensure dataset is downloaded
    dataset_id = "vipoooool/new-plant-diseases-dataset"
    dataset_root = ensure_dataset(dataset_id, root)

    # Paths
    dataset_root = os.path.join(dataset_root, "New Plant Diseases Dataset(Augmented)")
    train_dir = os.path.join(dataset_root, "New Plant Diseases Dataset(Augmented)/train")
    val_dir = os.path.join(dataset_root, "New Plant Diseases Dataset(Augmented)/valid")
    test_dir = os.path.join(root, "test")

    # Datasets
    train_ds = ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds = ImageFolder(val_dir, transform=get_transforms(train=False))
    test_ds = ImageFolder(test_dir, transform=get_transforms(train=False))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
