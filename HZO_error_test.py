import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import numpy as np

# 设置学术论文级绘图风格
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "grid.alpha": 0.3
})

# --- 1. 灵活的组件定义 ---

class ResBlock(nn.Module):
    def __init__(self, channels, act_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            act_fn()
        )
    def forward(self, x):
        return x + self.conv(x)

class FlexibleResNet(nn.Module):
    def __init__(self, depth, act_fn=nn.GELU, hidden=8):
        super().__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            act_fn()
        ))
        for _ in range(depth - 2):
            self.layer_list.append(ResBlock(hidden, act_fn))
        self.layer_list.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden, 10, bias=False)
        ))

    def forward(self, x):
        for layer in self.layer_list: x = layer(x)
        return x

# --- 2. HZO 引擎 (Dtype 兼容版) ---

class HZO_Engine:
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon
        self.grads = [None] * len(model.layer_list)

    def _perturb_activation(self, layers, x, target_grad):
        orig_shape = x.shape
        N, D = x.size(0), x.reshape(x.size(0), -1).size(1)
        flat_x = x.reshape(N, -1)
        grad_input_flat = torch.zeros_like(flat_x)
        eps = self.epsilon
        
        with torch.no_grad():
            for i in range(D):
                p = torch.zeros_like(flat_x); p[:, i] = eps
                # 核心：根据输入类型计算差分
                out_p = (flat_x + p).view(orig_shape)
                for l in layers: out_p = l(out_p)
                out_m = (flat_x - p).view(orig_shape)
                for l in layers: out_m = l(out_m)
                
                # 模拟不同精度下的减法误差
                diff = (out_p - out_m) / (2 * eps)
                grad_input_flat[:, i] = torch.sum(diff.reshape(N, -1) * target_grad.reshape(N, -1), dim=1)
        return grad_input_flat.view(orig_shape)

    def estimate(self, x, y_pred, y_target_oh):
        probs = torch.softmax(y_pred, dim=1)
        final_grad = (probs - y_target_oh) / x.size(0)
        
        # 递归分治逻辑
        def divide_conquer(layers, x_in, t_grad, idx):
            n = len(layers)
            if n == 1:
                self.grads[idx] = t_grad
                return
            mid = n // 2
            with torch.no_grad():
                mid_in = x_in
                for l in layers[:mid]: mid_in = l(mid_in)
            mid_t_grad = self._perturb_activation(layers[mid:], mid_in, t_grad)
            divide_conquer(layers[mid:], mid_in, t_grad, idx + mid)
            divide_conquer(layers[:mid], x_in, mid_t_grad, idx)

        divide_conquer(self.model.layer_list, x, final_grad, 0)
        
        z = x.clone().detach().requires_grad_(True)
        for i, layer in enumerate(self.model.layer_list):
            z_next = layer(z)
            if self.grads[i] is not None: z_next.backward(self.grads[i], retain_graph=False)
            z = z_next.detach().requires_grad_(True)

# --- 3. 实验一：数值精度对比 (Precision Experiment) ---

def run_precision_exp():
    depths = [16, 32, 64]
    dtypes = [torch.float16, torch.float32, torch.float64]
    dtype_labels = ["FP16", "FP32", "FP64 (Double)"]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    results = {d: [] for d in dtype_labels}
    
    for dtype, label in zip(dtypes, dtype_labels):
        for L in depths:
            print(f"Testing {label} at Depth {L}...")
            # 统一设备和数据
            data = torch.randn(2, 3, 32, 32).to(dtype)
            target = torch.randint(0, 10, (2,))
            target_oh = torch.nn.functional.one_hot(target, 10).to(dtype)
            
            model = FlexibleResNet(depth=L, hidden=4).to(dtype).eval()
            model_hzo = copy.deepcopy(model)
            
            # BP
            model.zero_grad()
            torch.nn.functional.cross_entropy(model(data), target).backward()
            g_bp = next(model.layer_list[0][0].parameters()).grad.view(-1)
            
            # HZO
            model_hzo.zero_grad()
            engine = HZO_Engine(model_hzo, epsilon=1e-3 if dtype != torch.float16 else 1e-2)
            engine.estimate(data, model_hzo(data), target_oh)
            g_hzo = next(model_hzo.layer_list[0][0].parameters()).grad.view(-1)
            
            sim = torch.nn.functional.cosine_similarity(g_bp, g_hzo, dim=0).item()
            results[label].append(sim)

    # 作图
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(dtype_labels):
        plt.plot(depths, results[label], marker='o', color=colors[i], label=label)
    
    plt.xlabel("Network Depth ($L$)")
    plt.ylabel("Gradient Cosine Similarity ($\\rho$)")
    plt.title("Numerical Precision Impact on HZO Fidelity")
    plt.ylim(-0.1, 1.1); plt.xticks(depths); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig('precision_comparison.png', dpi=300)
    print("Precision plot saved.")

if __name__ == "__main__":
    run_precision_exp()
