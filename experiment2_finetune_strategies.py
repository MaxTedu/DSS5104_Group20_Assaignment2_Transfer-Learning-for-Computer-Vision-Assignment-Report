import torch
import os
import json
from dataset import get_dataloaders
from models import get_model
from utils import train_model, validate, save_training_curves

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建保存目录
os.makedirs('results/finetune_strategies', exist_ok=True)

# 第一个实验中表现最好的模型
BEST_MODEL = 'vit_b_16'
n_classes = 9  # PathMNIST的类别数

strategies = [
    {
        'name': 'feature_extraction',
        'description': 'Freeze backbone, only train classifier head',
        'freeze_backbone': True,
        'augment': True,
        'lr': 1e-4
    },
    {
        'name': 'full_finetune',
        'description': 'Finetune entire network with small LR',
        'freeze_backbone': False,
        'augment': True,
        'lr': 3e-5
    },
    {
        'name': 'full_finetune_no_augment',
        'description': 'Finetune entire network without data augmentation',
        'freeze_backbone': False,
        'augment': False,
        'lr': 3e-5
    }
]

results = []

for strategy in strategies:
    print(f"\n{'='*50}")
    print(f"Running strategy: {strategy['name']}")
    print(f"Description: {strategy['description']}")
    print('='*50)
    
    # 加载对应增强的数据
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=32, 
        augment=strategy['augment']
    )
    
    # 初始化模型
    model = get_model(
        BEST_MODEL, 
        n_classes, 
        pretrained=True, 
        freeze_backbone=strategy['freeze_backbone']
    ).to(device)
    
    # 训练模型
    save_path = f'results/finetune_strategies/{BEST_MODEL}_{strategy["name"]}_best.pth'
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=20, 
        lr=strategy['lr'], 
        device=device, 
        save_path=save_path
    )
    
    # 保存训练曲线
    save_training_curves(history, f'results/finetune_strategies/{BEST_MODEL}_{strategy["name"]}_curves.png')
    
    # 测试集准确率
    test_loss, test_acc = validate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 计算过拟合程度
    final_train_acc = history[1][-1]
    overfit_gap = final_train_acc - test_acc
    print(f"Overfit gap: {overfit_gap:.2f}%")
    
    # 保存结果
    results.append({
        'strategy': strategy['name'],
        'description': strategy['description'],
        'test_acc': float(f"{test_acc:.2f}"),
        'final_train_acc': float(f"{final_train_acc:.2f}"),
        'overfit_gap': float(f"{overfit_gap:.2f}"),
        'train_losses': [float(f"{x:.4f}") for x in history[0]],
        'train_accs': [float(f"{x:.2f}") for x in history[1]],
        'val_losses': [float(f"{x:.4f}") for x in history[2]],
        'val_accs': [float(f"{x:.2f}") for x in history[3]]
    })

# 保存所有结果
with open('results/finetune_strategies/strategy_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# 打印对比表
print("\n" + "="*80)
print("Finetune Strategy Comparison Results:")
print("-"*80)
print(f"{'Strategy':<30} {'Test Acc (%)':<15} {'Train Acc (%)':<15} {'Overfit Gap (%)':<15}")
print("-"*80)
for res in results:
    print(f"{res['strategy']:<30} {res['test_acc']:<15} {res['final_train_acc']:<15} {res['overfit_gap']:<15}")
print("="*80)
