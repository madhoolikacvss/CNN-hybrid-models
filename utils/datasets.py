import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_transforms(train=True, img_size=224):
    """Returns the appropriate transformations for training or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def get_newplant_loaders(base_dir="/kaggle/input/new-plant-diseases-dataset", batch_size=32):
    """
    Returns train/val/test loaders for the New Plant Diseases Dataset (Augmented).
    Expects structure like:
        base_dir/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
        base_dir/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid
        base_dir/test
    """
    base_dataset_path = os.path.join(base_dir,
        "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)"
    )
    train_dir = os.path.join(base_dataset_path, "train")
    val_dir   = os.path.join(base_dataset_path, "valid")
    test_dir  = os.path.join(base_dir, "test")

    print(f"Looking for train data in: {train_dir}")
    print(f"Looking for validation data in: {val_dir}")
    print(f"Looking for test data in: {test_dir}")

    # Datasets
    train_ds = ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds   = ImageFolder(val_dir, transform=get_transforms(train=False))
    test_ds  = ImageFolder(test_dir, transform=get_transforms(train=False))

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_ds.classes
