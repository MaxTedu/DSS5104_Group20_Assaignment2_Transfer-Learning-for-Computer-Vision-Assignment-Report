import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from models import get_model
from utils import train_model, validate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建保存目录
os.makedirs('results/data_efficiency', exist_ok=True)

# 选择最优模型
BEST_MODEL = 'vit_b_16'
n_classes = 9
data_fractions = [1.0, 0.1, 0.05]  # 精简为3个最有代表性的梯度，足够看出趋势
seeds = [42]  # 只跑1个随机种子，降低计算量

results = {
    'pretrained': {str(f): [] for f in data_fractions},
    'scratch': {str(f): [] for f in data_fractions}
}

for fraction in data_fractions:
    print(f"\n{'='*60}")
    print(f"Running experiments with {fraction*100:.0f}% of training data")
    print('='*60)
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # 加载数据
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            batch_size=32, 
            data_fraction=fraction,
            augment=True
        )
        
        # 训练预训练模型
        print("Training pretrained model...")
        model_pretrained = get_model(BEST_MODEL, n_classes, pretrained=True).to(device)
        save_path_pre = f'results/data_efficiency/{BEST_MODEL}_pre_{fraction*100:.0f}_{seed}.pth'
        model_pretrained, _ = train_model(
            model_pretrained, 
            train_loader, 
            val_loader, 
            epochs=12,  # 降低epoch数量，加快训练
            lr=3e-5, 
            device=device, 
            save_path=save_path_pre
        )
        _, test_acc_pre = validate(model_pretrained, test_loader, torch.nn.CrossEntropyLoss(), device)
        print(f"Pretrained Test Acc: {test_acc_pre:.2f}%")
        results['pretrained'][str(fraction)].append(test_acc_pre)
        
        # 训练从头开始的模型
        print("\nTraining from scratch model...")
        model_scratch = get_model(BEST_MODEL, n_classes, pretrained=False).to(device)
        save_path_scratch = f'results/data_efficiency/{BEST_MODEL}_scratch_{fraction*100:.0f}_{seed}.pth'
        model_scratch, _ = train_model(
            model_scratch, 
            train_loader, 
            val_loader, 
            epochs=15,  # 降低从头训练的epoch数量
            lr=1e-3,  # 从头训练学习率更大
            device=device, 
            save_path=save_path_scratch
        )
        _, test_acc_scratch = validate(model_scratch, test_loader, torch.nn.CrossEntropyLoss(), device)
        print(f"Scratch Test Acc: {test_acc_scratch:.2f}%")
        results['scratch'][str(fraction)].append(test_acc_scratch)

# 统计均值和标准差
summary = {
    'pretrained': [],
    'scratch': []
}
for fraction in data_fractions:
    pre_accs = results['pretrained'][str(fraction)]
    pre_mean = np.mean(pre_accs)
    pre_std = np.std(pre_accs)
    summary['pretrained'].append({
        'fraction': float(fraction),
        'mean_acc': float(f"{pre_mean:.2f}"),
        'std_acc': float(f"{pre_std:.2f}"),
        'all_accs': [float(f"{x:.2f}") for x in pre_accs]
    })
    
    scratch_accs = results['scratch'][str(fraction)]
    scratch_mean = np.mean(scratch_accs)
    scratch_std = np.std(scratch_accs)
    summary['scratch'].append({
        'fraction': float(fraction),
        'mean_acc': float(f"{scratch_mean:.2f}"),
        'std_acc': float(f"{scratch_std:.2f}"),
        'all_accs': [float(f"{x:.2f}") for x in scratch_accs]
    })

# 保存结果
with open('results/data_efficiency/efficiency_results.json', 'w') as f:
    json.dump(summary, f, indent=4)

# 绘制对比曲线
plt.figure(figsize=(8, 5))
x = [f*100 for f in data_fractions]

pre_means = [s['mean_acc'] for s in summary['pretrained']]
pre_stds = [s['std_acc'] for s in summary['pretrained']]
plt.errorbar(x, pre_means, yerr=pre_stds, marker='o', label='Pretrained', capsize=5)

scratch_means = [s['mean_acc'] for s in summary['scratch']]
scratch_stds = [s['std_acc'] for s in summary['scratch']]
plt.errorbar(x, scratch_means, yerr=scratch_stds, marker='s', label='From Scratch', capsize=5)

plt.xlabel('Training Data Percentage (%)')
plt.ylabel('Test Accuracy (%)')
plt.title('Data Efficiency: Pretrained vs From Scratch')
plt.xscale('log')
plt.xticks(x, [f"{int(f)}%" for f in x])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/data_efficiency/efficiency_curve.png')
plt.close()

# 打印结果表
print("\n" + "="*80)
print("Data Efficiency Results Summary:")
print("-"*80)
print(f"{'Data %':<10} {'Pretrained Mean (Std)':<30} {'From Scratch Mean (Std)':<30}")
print("-"*80)
for i, frac in enumerate(data_fractions):
    pre_str = f"{pre_means[i]:.2f} ± {pre_stds[i]:.2f}"
    scratch_str = f"{scratch_means[i]:.2f} ± {scratch_stds[i]:.2f}"
    print(f"{frac*100:<10.0f} {pre_str:<30} {scratch_str:<30}")
print("="*80)
