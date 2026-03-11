import torch
import torch.nn as nn
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(loader, leave=False):
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, device='cuda', save_path='best_model.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    patience = 5
    counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最优权重
    model.load_state_dict(torch.load(save_path))
    return model, (train_losses, train_accs, val_losses, val_accs)

def measure_inference_speed(model, input_size=(3, 224, 224), device='cuda', iterations=1000):
    model.eval()
    input_tensor = torch.randn(1, *input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    
    # 测速
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    avg_latency = total_time / iterations * 1000  # 转换为毫秒
    return avg_latency

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_curves(history, save_path='training_curves.png'):
    train_losses, train_accs, val_losses, val_accs = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
