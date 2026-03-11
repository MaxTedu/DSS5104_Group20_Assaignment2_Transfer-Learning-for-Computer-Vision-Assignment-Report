import torch
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from models import get_model
from medmnist import INFO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载数据
_, _, test_loader, n_classes = get_dataloaders(batch_size=1, augment=False)
class_names = INFO['pathmnist']['label']

# 加载训练好的最优模型（实验2里的full_finetune_no_augment是最好的）
model = get_model('vit_b_16', n_classes, pretrained=False).to(device)
model.load_state_dict(torch.load('results/finetune_strategies/vit_b_16_full_finetune_no_augment_best.pth', map_location=device))
model.eval()

# 找出前几个错误分类的图片
error_samples = []
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        if len(error_samples) >= 3:
            break
        inputs = inputs.to(device)
        targets = targets.squeeze().long().to(device)
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        if predicted.item() != targets.item():
            # 反归一化图片用于显示
            img = inputs[0].cpu().numpy().transpose(1, 2, 0)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = std * img + mean
            img = img.clip(0, 1)
            
            error_samples.append({
                'image': img,
                'true_label': class_names[str(targets.item())],
                'pred_label': class_names[str(predicted.item())]
            })
            print(f"Found error sample {len(error_samples)}")

# 显示并保存错误样本
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, sample in enumerate(error_samples):
    axes[i].imshow(sample['image'])
    axes[i].set_title(f"True: {sample['true_label']}\nPred: {sample['pred_label']}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('results/error_samples.png', dpi=150, bbox_inches='tight')
print(f"Saved error samples to results/error_samples.png")
plt.show()
