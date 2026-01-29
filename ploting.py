import json
import matplotlib.pyplot as plt
import numpy as np

def plot_from_json(json_path, save_path):
    # 1. 加载数据
    with open(json_path, 'r') as f:
        history = json.load(f)

    # 2. 提取并计算维度
    # test_acc 是按 epoch 记录的，以此为基准确定 epoch 总数
    test_acc = history['test_acc']
    epoch_sim = history['epoch_sim']
    num_epochs = len(test_acc)
    
    batch_loss = np.array(history['batch_loss'])
    batch_acc = np.array(history['batch_acc'])
    
    # 计算每个 epoch 包含多少个 batch
    batches_per_epoch = len(batch_loss) // num_epochs
    
    # 对训练数据进行 Epoch 级别的平均化处理
    epoch_losses = []
    epoch_train_accs = []
    for i in range(num_epochs):
        start = i * batches_per_epoch
        end = (i + 1) * batches_per_epoch
        epoch_losses.append(np.mean(batch_loss[start:end]))
        epoch_train_accs.append(np.mean(batch_acc[start:end]) * 100) # 转为百分比

    epochs_range = range(1, num_epochs + 1)

    # 3. 开始绘图 (3 行 1 列)
    fig, axes = plt.subplots(3, 1, figsize=(10, 18), dpi=300)
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    # --- 第一张图: Training Loss & Accuracy (双 Y 轴) ---
    ax1_loss = axes[0]
    ax1_acc = ax1_loss.twinx()
    
    # 绘制 Loss (左轴 - 红色)
    ln1 = ax1_loss.plot(epochs_range, epoch_losses, color='#E74C3C', marker='o', 
                        markersize=4, label='Training Loss', linewidth=2)
    ax1_loss.set_ylabel('Cross Entropy Loss', color='#E74C3C', fontsize=13, fontweight='bold')
    ax1_loss.tick_params(axis='y', labelcolor='#E74C3C')
    
    # 绘制 Accuracy (右轴 - 蓝色)
    ln2 = ax1_acc.plot(epochs_range, epoch_train_accs, color='#3498DB', marker='s', 
                       markersize=4, label='Training Accuracy', linewidth=2)
    ax1_acc.set_ylabel('Training Accuracy (%)', color='#3498DB', fontsize=13, fontweight='bold')
    ax1_acc.tick_params(axis='y', labelcolor='#3498DB')
    
    ax1_loss.set_title('Training Convergence', fontsize=15, fontweight='bold', pad=15)
    ax1_loss.set_xlabel('Epoch', fontsize=12)
    
    # 合并两个坐标轴的图例
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1_loss.legend(lns, labs, loc='center right', frameon=True, shadow=True)
    ax1_loss.grid(True, linestyle='--', alpha=0.4)

    # --- 第二张图: Validation Accuracy ---
    axes[1].plot(epochs_range, test_acc, color='#8E44AD', marker='D', 
                 markersize=5, linewidth=2, label='Validation Accuracy')
    axes[1].set_title('Generalization Performance', fontsize=15, fontweight='bold', pad=15)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(np.arange(0, num_epochs + 1, max(1, num_epochs // 10)))
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(loc='lower right')

    # --- 第三张图: Gradient Fidelity (余弦相似度) ---
    axes[2].plot(epochs_range, epoch_sim, color='#27AE60', marker='^', 
                 markersize=5, linewidth=2, label=r'Cosine Similarity ($\rho$)')
    # 加上 1.0 的参考线
    axes[2].axhline(y=1.0, color='black', linestyle=':', alpha=0.6, label='Ideal Alignment')
    
    axes[2].set_title('Cosine Similarity (HZO vs BP)', fontsize=15, fontweight='bold', pad=15)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
    
    # 动态调整相似度的 Y 轴范围，防止死区过大
    ymin = min(epoch_sim) - 0.05
    axes[2].set_ylim([max(-1.0, ymin), 1.05]) 
    
    axes[2].set_xticks(np.arange(0, num_epochs + 1, max(1, num_epochs // 10)))
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Academic visualization generated: {save_path}")

if __name__ == "__main__":
    # 确保文件名与你保存的 json 一致
    plot_from_json('training_history.json', 'hzo_cifar10_final_report.png')