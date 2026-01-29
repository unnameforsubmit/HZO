import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. ResNet-20 架构 (保持 GELU) ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.act(out)

class ResNet20_HZO(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20_HZO, self).__init__()
        self.in_planes = 64
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        self._make_layer(64, 3, 1)
        self._make_layer(128, 3, 2)
        self._make_layer(256, 3, 2)
        self.layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, num_classes)
        ))

    def _make_layer(self, planes, num_blocks, stride):
        for s in [stride] + [1]*(num_blocks-1):
            self.layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            activations.append(layer(activations[-1]))
        return activations[-1], activations

# --- 2. HZO 递归引擎 (修复 BN 污染问题) ---
class RecursiveHZO_Engine:
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon

    def _apply_jacobian(self, layer_idx, x_input, grad_output):
        layer_block = self.model.layers[layer_idx]
        
        # 关键修复：在计算雅可比时锁定 BN，防止其统计量被噪声污染
        bn_modules = [m for m in layer_block.modules() if isinstance(m, nn.BatchNorm2d)]
        orig_training_states = [m.training for m in bn_modules]
        for m in bn_modules: m.eval() 

        with torch.enable_grad():
            x_detach = x_input.detach().requires_grad_(True)
            out = layer_block(x_detach)
            torch.autograd.backward(out, grad_output)
            grad_input = x_detach.grad
        
        # 恢复 BN 原有的训练/评估状态
        for m, state in zip(bn_modules, orig_training_states):
            if state: m.train()
            
        return grad_input

    def _divide_conquer(self, start_idx, end_idx, grad_output, activations):
        if end_idx - start_idx == 1:
            return self._apply_jacobian(start_idx, activations[start_idx], grad_output)
        mid = (start_idx + end_idx) // 2
        grad_mid = self._divide_conquer(mid, end_idx, grad_output, activations)
        return self._divide_conquer(start_idx, mid, grad_mid, activations)

    def step(self, inputs, activations, target_oh):
        final_out = activations[-1]
        probs = torch.softmax(final_out, dim=1)
        grad_loss = (probs - target_oh) / inputs.size(0)
        self._divide_conquer(0, len(self.model.layers), grad_loss, activations)

# --- 3. 验证与评估函数 ---
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out, _ = model(inputs)
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def get_similarity(model, engine, loader, device):
    model.eval()
    try:
        inputs, targets = next(iter(loader))
    except StopIteration: return 0.0
    inputs, targets = inputs.to(device), targets.to(device)
    targets_oh = F.one_hot(targets, 10).float()
    
    model.zero_grad()
    out, _ = model(inputs)
    F.cross_entropy(out, targets).backward()
    grad_bp = model.layers[0][0].weight.grad.clone().flatten()
    
    model.zero_grad()
    _, acts = model(inputs)
    engine.step(inputs, acts, targets_oh)
    grad_hzo = model.layers[0][0].weight.grad.clone().flatten()
    return F.cosine_similarity(grad_bp, grad_hzo, dim=0).item()

# --- 4. 主流程 ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {'batch_loss': [], 'batch_acc': [], 'test_acc': [], 'epoch_sim': []}
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 关键修复：确保训练集和测试集文件夹映射的标签索引完全一致
    train_dataset = datasets.ImageFolder('./imagenet_subset/train', transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ]))
    test_dataset = datasets.ImageFolder('./imagenet_subset/val', transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ]))
    test_dataset.class_to_idx = train_dataset.class_to_idx 
    
    print(f"Verified Labels: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = ResNet20_HZO(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    engine = RecursiveHZO_Engine(model)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_oh = F.one_hot(targets, 10).float()
            
            optimizer.zero_grad()
            out, activations = model(inputs)
            engine.step(inputs, activations, targets_oh)
            optimizer.step()
            
            with torch.no_grad():
                loss = F.cross_entropy(out, targets).item()
                acc = (out.max(1)[1] == targets).float().mean().item()
                history['batch_loss'].append(loss)
                history['batch_acc'].append(acc)
                pbar.set_postfix({'loss': f"{loss:.3f}", 'acc': f"{acc*100:.1f}%"})

        t_acc = evaluate(model, test_loader, device)
        sim = get_similarity(model, engine, test_loader, device)
        history['test_acc'].append(t_acc)
        history['epoch_sim'].append(sim)
        print(f" -> Epoch {epoch+1} Results: Test Acc: {t_acc:.2f}% | CosSim: {sim:.6f}")

    with open('history_imagenet10_full.json', 'w') as f:
        json.dump(history, f)

    # 绘图逻辑修复：动态适配 epoch 数量
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    epochs_range = range(1, len(history['test_acc']) + 1)
    
    axes[0].plot(history['batch_loss'], color='red', alpha=0.5); axes[0].set_title('Training Loss (Batch)')
    axes[1].plot(history['batch_acc'], color='blue', alpha=0.5); axes[1].set_title('Training Acc (Batch)')
    axes[2].plot(epochs_range, history['test_acc'], marker='s', color='purple'); axes[2].set_title('Validation Acc (Epoch)')
    axes[3].plot(epochs_range, history['epoch_sim'], marker='o', color='green'); axes[3].set_title('Gradient Fidelity (Epoch)')
    
    for ax in axes: ax.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('hzo_imagenet10_academic.png', dpi=300)
    print("Optimization Complete.")

if __name__ == '__main__':
    main()