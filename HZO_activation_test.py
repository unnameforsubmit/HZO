import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.3,
    "figure.dpi": 300
})

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
                out_p = (flat_x + p).view(orig_shape)
                for l in layers: out_p = l(out_p)
                out_m = (flat_x - p).view(orig_shape)
                for l in layers: out_m = l(out_m)
                diff = (out_p.double() - out_m.double()) / (2 * eps)
                grad_input_flat[:, i] = torch.sum(diff.float().reshape(N, -1) * target_grad.reshape(N, -1), dim=1)
        return grad_input_flat.view(orig_shape)

    def estimate(self, x, y_pred, y_target_oh):
        probs = torch.softmax(y_pred, dim=1)
        final_grad = (probs - y_target_oh) / x.size(0)
        
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

def run_activation_exp_enhanced():
    acts = [nn.ReLU, nn.GELU]
    labels = ["ReLU", "GELU"]
    depth = 32
    trials = 5
    
    data_list = []
    
    print(f"Starting Activation Fidelity Comparison (Depth={depth})...")
    
    for act_fn, label in zip(acts, labels):
        for t in range(trials):
            data = torch.randn(2, 3, 32, 32)
            target = torch.randint(0, 10, (2,))
            target_oh = torch.nn.functional.one_hot(target, 10).float()
            
            model = FlexibleResNet(depth=depth, act_fn=act_fn, hidden=8).eval()
            model_hzo = copy.deepcopy(model)
            
            model.zero_grad()
            torch.nn.functional.cross_entropy(model(data), target).backward()
            g_bp = next(model.layer_list[0][0].parameters()).grad.view(-1)
            
            engine = HZO_Engine(model_hzo, epsilon=1e-3)
            engine.estimate(data, model_hzo(data), target_oh)
            g_hzo = next(model_hzo.layer_list[0][0].parameters()).grad.view(-1)
            
            sim = torch.nn.functional.cosine_similarity(g_bp, g_hzo, dim=0).item()
            data_list.append({"Activation": label, "Similarity": sim})
            print(f" Trial {t+1}/{trials} | {label}: {sim:.4f}")

    df = pd.DataFrame(data_list)

    plt.figure(figsize=(7, 6))
    
    palette = {"ReLU": "#7F8C8D", "GELU": "#2980B9"} 
    ax = sns.boxplot(x="Activation", y="Similarity", data=df, 
                     palette=palette, width=0.5, showfliers=False,
                     linewidth=1.2, boxprops=dict(alpha=0.2))
    
    sns.stripplot(x="Activation", y="Similarity", data=df, 
                  palette=palette, size=6, jitter=True, alpha=0.8, dodge=False)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='ReLU (Zero-Order Bias)',
                              markerfacecolor=palette["ReLU"], markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='GELU (Higher Fidelity)',
                              markerfacecolor=palette["GELU"], markersize=10)]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True)

    plt.ylabel("Gradient Cosine Similarity ($\\rho$)")
    plt.xlabel("Activation Function Type")
    plt.title(f"Impact of Activation Smoothness on HZO Fidelity\n(ResNet Depth = {depth})", pad=15)
    plt.ylim(0.9, 1.02)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('hzo_activation_fidelity_comparison.png')
    plt.show()
    print("Enhanced activation comparison plot saved.")

if __name__ == "__main__":
    run_activation_exp_enhanced()