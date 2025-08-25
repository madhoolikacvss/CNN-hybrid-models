import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import kagglehub

def ensure_dataset(dataset_id: str, local_path: str):
    if not os.path.exists(local_path):
        print(f"Downloading {dataset_id} to {local_path}...")
        path = kagglehub.dataset_download(dataset_id)
        print("Downloaded to:", path)
        return path
    else:
        print(f"Using existing dataset at {local_path}")
        return local_path

def get_transforms(train=True):
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

def get_newplant_loaders(root="kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/train", batch_size=32):
    ensure_dataset("vipoooool/new-plant-diseases-dataset", root)
    train_ds = ImageFolder(f"{root}/train", transform=get_transforms(train=True))
    val_ds = ImageFolder(f"{root}/valid", transform=get_transforms(train=False))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4))

def get_plantvillage_loader(root="data/plantvillage", batch_size=32):
    ensure_dataset("emmarex/plantdisease", root)
    test_ds = ImageFolder(root, transform=get_transforms(train=False))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
