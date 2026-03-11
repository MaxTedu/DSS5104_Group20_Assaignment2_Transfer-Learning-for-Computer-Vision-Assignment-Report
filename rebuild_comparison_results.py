import torch
import os
import json
from dataset import get_dataloaders
from models import get_model
from utils import validate, count_parameters, measure_inference_speed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载测试数据
_, _, test_loader, n_classes = get_dataloaders(batch_size=32, augment=True)
criterion = torch.nn.CrossEntropyLoss()

models_to_test = ['resnet50', 'efficientnet_b0', 'vit_b_16']
results = []
result_dir = 'results/architecture_comparison'

for model_name in models_to_test:
    print(f"\n{'='*50}")
    print(f"Processing {model_name}...")
    print('='*50)
    
    # 初始化模型
    model = get_model(model_name, n_classes, pretrained=False).to(device)
    
    # 加载训练好的权重
    weight_path = os.path.join(result_dir, f'{model_name}_best.pth')
    if not os.path.exists(weight_path):
        print(f"Warning: {weight_path} not found, skipping...")
        continue
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    # 统计参数量
    param_count = count_parameters(model) / 1e6  # 转换为M
    print(f"Parameters: {param_count:.2f} M")
    
    # 测试集准确率
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # 测量推理速度
    latency = measure_inference_speed(model, device=device)
    print(f"Average Inference Latency: {latency:.2f} ms")
    
    # 统计模型文件大小
    model_size = os.path.getsize(weight_path) / (1024 * 1024)  # 转换为MB
    print(f"Model Size: {model_size:.2f} MB")
    
    # 保存结果
    results.append({
        'model_name': model_name,
        'parameters_m': float(f"{param_count:.2f}"),
        'test_acc': float(f"{test_acc:.2f}"),
        'latency_ms': float(f"{latency:.2f}"),
        'model_size_mb': float(f"{model_size:.2f}")
    })

# 保存完整结果
with open(os.path.join(result_dir, 'full_comparison_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# 打印完整对比表
print("\n" + "="*80)
print("Full Architecture Comparison Results:")
print("-"*80)
print(f"{'Model':<20} {'Params (M)':<12} {'Test Acc (%)':<15} {'Latency (ms)':<15} {'Size (MB)':<10}")
print("-"*80)
for res in results:
    print(f"{res['model_name']:<20} {res['parameters_m']:<12} {res['test_acc']:<15} {res['latency_ms']:<15} {res['model_size_mb']:<10}")
print("="*80)
print(f"\nFull results saved to {os.path.join(result_dir, 'full_comparison_results.json')}")
