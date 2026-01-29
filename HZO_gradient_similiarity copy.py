import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import numpy as np

# Set global styles for paper-quality plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "grid.alpha": 0.3
})

# --- 1. Model Definitions ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    def forward(self, x):
        return x + self.conv(x) # Residual Connection

class PlainBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x) # No Residual Connection

class DeepNet(nn.Module):
    def __init__(self, depth, hidden=8, use_res=True):
        super().__init__()
        self.layer_list = nn.ModuleList()
        # Input Layer
        self.layer_list.append(nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        ))
        # Hidden Layers
        for _ in range(depth - 2):
            if use_res:
                self.layer_list.append(ResBlock(hidden))
            else:
                self.layer_list.append(PlainBlock(hidden))
        # Output Layer
        self.layer_list.append(nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden, 10, bias=False)
        ))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

# --- 2. Gradient Engines ---

class HZO_Engine:
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon
        self.grads = [None] * len(model.layer_list)

    def _perturb_activation(self, layers, x, target_grad):
        orig_shape = x.shape
        N = x.size(0)
        flat_x = x.reshape(N, -1)
        D = flat_x.size(1)
        grad_input_flat = torch.zeros_like(flat_x)
        eps = self.epsilon
        
        with torch.no_grad():
            # For speed in simulation, we sample a subset of neurons if D is very large
            # But for 1.0 sim, we try to be precise.
            for i in range(D):
                p = torch.zeros_like(flat_x); p[:, i] = eps
                # Difference with Double Precision
                out_p = (flat_x + p).view(orig_shape)
                for l in layers: out_p = l(out_p)
                out_m = (flat_x - p).view(orig_shape)
                for l in layers: out_m = l(out_m)
                
                diff = (out_p.double() - out_m.double()) / (2 * eps)
                grad_input_flat[:, i] = torch.sum(diff.float().reshape(N, -1) * target_grad.reshape(N, -1), dim=1)
        return grad_input_flat.view(orig_shape)

    def _divide_conquer(self, layers, x, target_grad, index):
        n = len(layers)
        if n == 1:
            self.grads[index] = target_grad
            return
        mid = n // 2
        with torch.no_grad():
            mid_input = x
            for l in layers[:mid]: mid_input = l(mid_input)
        
        mid_target_grad = self._perturb_activation(layers[mid:], mid_input, target_grad)
        self._divide_conquer(layers[mid:], mid_input, target_grad, index + mid)
        self._divide_conquer(layers[:mid], x, mid_target_grad, index)

    def estimate(self, x, y_pred, y_target_oh):
        probs = torch.softmax(y_pred, dim=1)
        final_grad = (probs - y_target_oh) / x.size(0)
        self._divide_conquer(self.model.layer_list, x, final_grad, index=0)
        
        # Inject gradients
        z = x.clone().detach().requires_grad_(True)
        for i, layer in enumerate(self.model.layer_list):
            z_next = layer(z)
            if self.grads[i] is not None:
                z_next.backward(self.grads[i], retain_graph=False)
            z = z_next.detach().requires_grad_(True)

class Standard_ZO_Engine:
    """Standard ZO: Non-hierarchical activation perturbation (End-to-end)"""
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon

    def estimate(self, x, y_pred, y_target_oh):
        probs = torch.softmax(y_pred, dim=1)
        final_grad = (probs - y_target_oh) / x.size(0)
        
        # Standard ZO treats the whole network as one block to estimate the first layer gradient
        # It estimates dL/da_1 where a_1 is the output of layer 0
        with torch.no_grad():
            a0 = self.model.layer_list[0](x)
        
        # Helper to perturb and pass through remaining layers
        orig_shape = a0.shape
        flat_a = a0.reshape(a0.size(0), -1)
        D = flat_a.size(1)
        grad_a0 = torch.zeros_like(flat_a)
        eps = self.epsilon
        
        with torch.no_grad():
            # Sample subset for speed if needed, but here we do full for precision comparison
            # limit D to 64 for standard ZO speed in this script if needed, but let's try full
            for i in range(D):
                p = torch.zeros_like(flat_a); p[:, i] = eps
                out_p = (flat_a + p).view(orig_shape)
                for l in self.model.layer_list[1:]: out_p = l(out_p)
                out_m = (flat_a - p).view(orig_shape)
                for l in self.model.layer_list[1:]: out_m = l(out_m)
                
                diff = (out_p.double() - out_m.double()) / (2 * eps)
                grad_a0[:, i] = torch.sum(diff.float().reshape(x.size(0), -1) * final_grad.reshape(x.size(0), -1), dim=1)
        
        # Backward locally to get weight grads
        z = x.clone().detach().requires_grad_(True)
        self.model.layer_list[0](z).backward(grad_a0.view(orig_shape))

# --- 3. Main Testing Loop ---

depths = [16, 32, 64]
results = {
    "HZO_Res": [],
    "HZO_plain": [],
    "ZO_Res": []
}

device = torch.device('cpu')
batch_size = 2
data = torch.randn(batch_size, 3, 32, 32).to(device)
target = torch.randint(0, 10, (batch_size,)).to(device)
target_oh = torch.nn.functional.one_hot(target, 10).float().to(device)

for L in depths:
    print(f"Running Depth {L}...")
    
    # 1. HZO_Res
    m_bp = DeepNet(depth=L, hidden=4, use_res=True).to(device).eval()
    m_hzo = copy.deepcopy(m_bp)
    
    m_bp.zero_grad()
    torch.nn.functional.cross_entropy(m_bp(data), target).backward()
    g_bp = next(m_bp.layer_list[0][0].parameters()).grad.view(-1)
    
    m_hzo.zero_grad()
    hzo = HZO_Engine(m_hzo, epsilon=1e-3)
    hzo.estimate(data, m_hzo(data), target_oh)
    g_hzo = next(m_hzo.layer_list[0][0].parameters()).grad.view(-1)
    results["HZO_Res"].append(torch.nn.functional.cosine_similarity(g_bp, g_hzo, dim=0).item())

    # 2. HZO_plain
    m_bp_p = DeepNet(depth=L, hidden=4, use_res=False).to(device).eval()
    m_hzo_p = copy.deepcopy(m_bp_p)
    
    m_bp_p.zero_grad()
    torch.nn.functional.cross_entropy(m_bp_p(data), target).backward()
    g_bp_p = next(m_bp_p.layer_list[0][0].parameters()).grad.view(-1)
    
    m_hzo_p.zero_grad()
    hzo_p = HZO_Engine(m_hzo_p, epsilon=1e-3)
    hzo_p.estimate(data, m_hzo_p(data), target_oh)
    g_hzo_p = next(m_hzo_p.layer_list[0][0].parameters()).grad.view(-1)
    results["HZO_plain"].append(torch.nn.functional.cosine_similarity(g_bp_p, g_hzo_p, dim=0).item())

    # 3. ZO_Res
    m_zo = copy.deepcopy(m_bp) # uses the same ResNet BP ground truth
    m_zo.zero_grad()
    zo_engine = Standard_ZO_Engine(m_zo, epsilon=1e-3)
    zo_engine.estimate(data, m_zo(data), target_oh)
    g_zo = next(m_zo.layer_list[0][0].parameters()).grad.view(-1)
    results["ZO_Res"].append(torch.nn.functional.cosine_similarity(g_bp, g_zo, dim=0).item())

# --- 4. Plotting ---

plt.figure(figsize=(8, 6))
plt.plot(depths, results["HZO_Res"], marker='s', color='royalblue', label='HZO (ResNet)')
plt.plot(depths, results["HZO_plain"], marker='o', color='darkorange', label='HZO (Plain CNN)')
plt.plot(depths, results["ZO_Res"], marker='^', color='forestgreen', linestyle='--', label='Standard ZO (ResNet)')

plt.xlabel("Network Depth ($L$)")
plt.ylabel("Gradient Cosine Similarity ($\\rho$)")
plt.title("Gradient Fidelity vs. Depth")
plt.ylim(-0.1, 1.1)
plt.xticks(depths)
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()

plt.savefig('fidelity_depth_comparison.png', dpi=300)
print("Plot saved as fidelity_depth_comparison.png")
print("Results:", results)