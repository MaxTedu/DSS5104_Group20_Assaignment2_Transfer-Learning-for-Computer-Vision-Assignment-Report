import torch
import torchvision.models as models
import torch.nn as nn

def get_model(model_name, n_classes, pretrained=True, freeze_backbone=False):
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
    
    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, n_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.heads.parameters():
                param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
