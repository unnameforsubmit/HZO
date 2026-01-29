import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CONFIG = {
    'width': 64,              
    'batch_size': 128,        
    'lr': 1e-3,               
    'weight_decay': 1e-2,     
    'epochs': 20,
    'hzo_epsilon': 1e-3,      
    'save_path': './hzo_resnet_ckpt.pth',
    'history_path': './training_history.json', 
    'plot_path': './hzo_training_plot.png'      
}

class PlainCNN_8L(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.layers = nn.ModuleList()
        in_c = 3
        for i in range(7):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_c, hidden, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU()
            ))
            in_c = hidden
        self.layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden, 10, bias=False)
        ))

    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            activations.append(layer(activations[-1]))
        return activations[-1], activations

class RecursiveHZO_Engine:
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon

    def _apply_jacobian(self, layer_idx, x_input, grad_output):
        layer_block = self.model.layers[layer_idx]
        conv_layer = layer_block[0]

        if not isinstance(conv_layer, nn.Conv2d):
            with torch.enable_grad():
                x_detach = x_input.detach().requires_grad_(True)
                out = layer_block(x_detach)
                torch.autograd.backward(out, grad_output)
                return x_detach.grad

        k, pad, stride = conv_layer.kernel_size[0], conv_layer.padding[0], conv_layer.stride[0]
        N, C, H, W = x_input.shape

        bn_stats = []
        for m in layer_block.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_stats.append(m.track_running_stats)
                m.track_running_stats = False

        grad_input = torch.zeros_like(x_input)
        sum_kernel = torch.ones(1, 1, k, k, device=x_input.device)

        with torch.no_grad():
            for kh in range(k):
                for kw in range(k):
                    mask = torch.zeros(1, 1, H, W, device=x_input.device)
                    mask[:, :, kh::k, kw::k] = 1.0
                    x_exp = x_input.repeat_interleave(C, dim=0)
                    eye = torch.eye(C, device=x_input.device).view(C, C, 1, 1).repeat(N, 1, 1, 1)
                    perturb = eye * mask * self.epsilon
                    
                    out_p = layer_block(x_exp + perturb)
                    out_m = layer_block(x_exp - perturb)
                    delta_y = (out_p - out_m) / (2 * self.epsilon)
                    
                    g_out_exp = grad_output.repeat_interleave(C, dim=0)
                    grad_map_sum = (g_out_exp * delta_y).sum(dim=1, keepdim=True)
                    grad_agg = F.conv2d(grad_map_sum, sum_kernel, stride=stride, padding=pad)
                    grad_input += grad_agg.view(N, C, H, W) * mask

        for i, m in enumerate([m for m in layer_block.modules() if isinstance(m, nn.BatchNorm2d)]):
            m.track_running_stats = bn_stats[i]
        
        with torch.enable_grad():
            x_detach = x_input.detach().requires_grad_(True)
            out = layer_block(x_detach)
            torch.autograd.backward(out, grad_output)
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

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out, _ = model(inputs)
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def get_similarity(model, engine, test_loader, device):
    model.eval()
    try:
        inputs, targets = next(iter(test_loader))
    except StopIteration: return 0.0
    inputs, targets = inputs.to(device), targets.to(device)
    targets_oh = F.one_hot(targets, 10).float()
    
    model.zero_grad()
    out, _ = model(inputs)
    F.cross_entropy(out, targets).backward()
    grad_bp = model.layers[0][0].weight.grad.clone().flatten()
    
    model.zero_grad()
    _, activations = model(inputs)
    engine.step(inputs, activations, targets_oh)
    grad_hzo = model.layers[0][0].weight.grad.clone().flatten()
    
    return F.cosine_similarity(grad_bp, grad_hzo, dim=0).item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    history = {
        'batch_loss': [],
        'batch_acc': [],
        'test_acc': [],
        'epoch_sim': []
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), 
                              batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=transform), 
                             batch_size=CONFIG['batch_size'], shuffle=False)

    model = PlainCNN_8L(hidden=CONFIG['width']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    engine = RecursiveHZO_Engine(model, epsilon=CONFIG['hzo_epsilon'])

    print(f"Training started (ReLU). Logging Loss/Acc per batch, Test Acc/Similarity per epoch.")

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=110)
        
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

        test_acc = evaluate(model, test_loader, device)
        sim = get_similarity(model, engine, test_loader, device)
        
        history['test_acc'].append(test_acc)
        history['epoch_sim'].append(sim)
        
        print(f" -> Epoch {epoch+1} Results: Test Acc: {test_acc:.2f}% | CosSim: {sim:.6f}")

    with open(CONFIG['history_path'], 'w') as f:
        json.dump(history, f)

    print("Plotting results...")
    fig, axes = plt.subplots(4, 1, figsize=(10, 18))
    
    axes[0].plot(history['batch_loss'], color='red', alpha=0.3)
    axes[0].set_title('Training Loss (Per Batch)')
    axes[0].grid(True)

    axes[1].plot(history['batch_acc'], color='blue', alpha=0.3)
    axes[1].set_title('Training Accuracy (Per Batch)')
    axes[1].grid(True)

    epochs_range = range(1, CONFIG['epochs'] + 1)
    axes[2].plot(epochs_range, history['test_acc'], marker='s', color='purple', linewidth=2)
    axes[2].set_title('Validation Accuracy (Per Epoch)')
    axes[2].set_xticks(epochs_range)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].grid(True)

    axes[3].plot(epochs_range, history['epoch_sim'], marker='o', color='green', linewidth=2)
    axes[3].set_title('Gradient Cosine Similarity (Per Epoch)')
    axes[3].set_xticks(epochs_range)
    axes[3].set_ylim([0.9, 1.01])
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(CONFIG['plot_path'])
    print(f"Visual report saved to {CONFIG['plot_path']}")

if __name__ == '__main__':
    main()