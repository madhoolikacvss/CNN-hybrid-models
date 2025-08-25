import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Define the directory where the dataset will be stored
DATA_DIR = '/content/CNN-hybrid-models/data'

def ensure_dataset(dataset_id='vipoooool/new-plant-diseases-dataset', local_path=DATA_DIR):
    """Ensures the dataset is downloaded and extracted."""
    dataset_zip_path = os.path.join(local_path, 'new-plant-diseases-dataset.zip')
    extracted_path = local_path

    # Check if the unzipped directory exists with the correct nesting
    if os.path.exists(os.path.join(extracted_path, 'New Plant Diseases Dataset(Augmented)', 'New Plant Diseases Dataset(Augmented)')):
        print(f"Using existing dataset at {local_path}")
        return local_path

    print(f"Downloading {dataset_id} to {local_path}...")
    api = KaggleApi()
    api.authenticate()

    # Create the local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Download the dataset
    api.dataset_download_files(dataset_id, path=local_path, unzip=False) # Download zip first

    print(f"Extracting dataset to {extracted_path}...")
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    print("Extraction complete.")

    return local_path # Return the local_path

def get_transforms(train=True):
    """Returns the appropriate transformations for training or validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_newplant_loaders(root=DATA_DIR, batch_size=32):
    """Returns data loaders for the New Plant Diseases Dataset."""
    dataset_path = ensure_dataset(local_path=root) # Call ensure_dataset and get the path, pass root

    # Construct the full path to the dataset subdirectories based on observed structure with nested directory
    base_dataset_path = os.path.join(dataset_path, 'New Plant Diseases Dataset(Augmented)','New Plant Diseases Dataset(Augmented)')
    train_dir = os.path.join(base_dataset_path, 'train')
    val_dir = os.path.join(base_dataset_path, 'valid')
    test_dir = os.path.join(dataset_path, 'test') # Include the test directory

    # Debugging print statements
    print(f"Looking for train data in: {train_dir}")
    print(f"Looking for validation data in: {val_dir}")
    print(f"Looking for test data in: {test_dir}")


    # Datasets
    train_ds = ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds = ImageFolder(val_dir, transform=get_transforms(train=False))
    test_ds = ImageFolder(test_dir, transform=get_transforms(train=False)) # Create test dataset


    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2) # Create test dataloader

    return train_loader, val_loader, test_loader # Return all three loaders
