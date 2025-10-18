import torch
import torchvision
import torchvision.transforms as transforms


def get_device() -> torch.device:
    '''
        suppport cuda, apple silicon, and cpu
    '''
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")



def load_model(batch_size=4):
    """
    Load CIFAR-10 dataset with optimized data pipeline and augmentation.
    
    Best practices applied:
    - Data augmentation for training set (RandomCrop, RandomHorizontalFlip)
    - Proper normalization with CIFAR-10 statistics
    - Optimized DataLoader settings (pin_memory, persistent_workers, num_workers)
    """
    
    # Training transform with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),     # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 statistics
    ])
    
    # Test transform without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # CIFAR-10 statistics
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Determine optimal num_workers based on device
    device = get_device()
    if device.type == 'cuda':
        num_workers = 4  # 4 workers per GPU
        pin_memory = True
    elif device.type == 'mps':
        num_workers = 4  # Apple Silicon handles this well
        pin_memory = True
    else:
        num_workers = 2
        pin_memory = False

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,  # Avoid worker restart overhead
        prefetch_factor=2 if num_workers > 0 else None  # Overlap data loading with training
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes
