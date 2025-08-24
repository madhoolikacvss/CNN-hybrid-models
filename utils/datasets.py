import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(20),
            T.GaussianBlur(kernel_size = 5, sigma=(0.1,2.0)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]) #using imagenet mean and std
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]) #using imagenet mean and std
        ])

def get_dataloaders(data_dir, batch_size=32, split=0.8):
    dataset = ImageFolder(data_dir, transform=get_transforms(train=True))
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4),
    )

def get_testloader(data_dir, batch_size=32):
    dataset = ImageFolder(data_dir, transform=get_transforms(train=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
