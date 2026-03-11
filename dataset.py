import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(dataset_name='pathmnist', batch_size=32, data_fraction=1.0, augment=True):
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    n_classes = len(info['label'])
    
    # 基础预处理
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 训练增强
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose(base_transform)
    
    val_test_transform = transforms.Compose(base_transform)
    
    # 加载数据集
    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=val_test_transform, download=True)
    test_dataset = DataClass(split='test', transform=val_test_transform, download=True)
    
    # 数据量缩减（用于数据效率实验）
    if data_fraction < 1.0:
        total_train = len(train_dataset)
        indices = np.random.choice(total_train, int(total_train * data_fraction), replace=False)
        train_dataset = Subset(train_dataset, indices)
    
    # 创建DataLoader，Windows下使用num_workers=0避免多进程问题
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, n_classes
