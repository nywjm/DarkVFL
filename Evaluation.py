import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

log_file_path = 'cifar1.txt'  

epochs = []
train_acc_dict = {}
test_acc_list = []
test_asr_list = []

best_epoch = 0
best_test_acc = 0.0
best_asr_at_best_epoch = 0.0
max_asr = 0.0
max_asr_epoch = 0
last_epoch = 0

with open(log_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    
    train_match = re.search(r'Epoch:\s+(\d+),\s+\d+/\d+:.*train main task accuracy:\s+([\d.]+)', line)
    if train_match:
        ep = int(train_match.group(1))
        train_acc_dict[ep] = float(train_match.group(2))

    test_match = re.search(r'=> Test Epoch:\s+(\d+),.*test main task accuracy:\s+([\d.]+),.*test asr:\s+([\d.]+)', line)
    if test_match:
        ep = int(test_match.group(1))
        test_acc = float(test_match.group(2))
        asr = float(test_match.group(3))

        epochs.append(ep)
        test_acc_list.append(test_acc)
        test_asr_list.append(asr)

        if asr > max_asr:
            max_asr = asr
            max_asr_epoch = ep

    best_match = re.search(
        r'=> End Epoch:\s+(\d+),.*best epoch:\s+(\d+),.*main task accuracy:\s+([\d.]+),.*test asr:\s+([\d.]+)', line)
    if best_match:
        last_epoch = int(best_match.group(1))
        best_epoch = int(best_match.group(2))
        best_test_acc = float(best_match.group(3))
        best_asr_at_best_epoch = float(best_match.group(4))

train_acc_list = [train_acc_dict[ep] for ep in epochs]

fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

ax1.plot(epochs, train_acc_list, label='training dataset main task accuracy', marker='o', linestyle='-', color='#1f77b4', linewidth=2,
         markersize=4)
ax1.plot(epochs, test_acc_list, label='testing dataset main task accuracy', marker='s', linestyle='--', color='#ff7f0e', linewidth=2,
         markersize=4)

ax1.plot(best_epoch, best_test_acc, marker='o', color='red', markersize=8)
ax1.annotate(f'best_test_accuracy: {best_test_acc}\nEpoch: {best_epoch}',
             xy=(best_epoch, best_test_acc),
             xytext=(best_epoch + 2, best_test_acc - 0.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10)
if last_epoch > 0 and last_epoch in epochs:
    
    last_train_acc = train_acc_dict.get(last_epoch, "no")
    
    last_epoch_idx = epochs.index(last_epoch)
    last_test_acc = test_acc_list[last_epoch_idx]
    last_test_asr = test_asr_list[last_epoch_idx]

ax1.set_ylim(0, 1.0)
ax1.set_xlabel('（Epoch）', fontsize=12)
ax1.set_ylabel('（Accuracy）', fontsize=12)
ax1.set_title('CIFAR10-DarkVFL', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

plt.tight_layout()
fig1.savefig('CIFAR10_DarkVFL_Accuracy.png')

fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=300)

ax2.plot(epochs, test_asr_list, label='testing datasetASR', marker='o', linestyle='-', color='#d62728', linewidth=2,
         markersize=4)

ax2.plot(max_asr_epoch, max_asr, marker='o', color='darkred', markersize=8)
ax2.annotate(f'Max ASR: {max_asr}\nEpoch: {max_asr_epoch}',
             xy=(max_asr_epoch, max_asr),
             xytext=(max_asr_epoch + 2, max_asr - 0.08),
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
             fontsize=10)

best_asr_val = test_asr_list[epochs.index(best_epoch)]
ax2.plot(best_epoch, best_asr_val, marker='o', color='orange', markersize=8)
ax2.annotate(f'Best ASR: {best_asr_val}\nEpoch: {best_epoch}',
             xy=(best_epoch, best_asr_val),
             xytext=(best_epoch + 2, best_asr_val - 0.12),
             arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
             fontsize=10)

early_stop_start = best_epoch + 1
ax2.axvspan(early_stop_start, early_stop_start+19, facecolor='lightgrey', alpha=0.5,
             label=f'early stop（Epoch{early_stop_start}-{early_stop_start+19}）')

ax2.set_ylim(0, 1.0)
ax2.set_xlabel('（Epoch）', fontsize=12)
ax2.set_ylabel('（ASR）', fontsize=12)
ax2.set_title('DarkVFL ASR with 10% poisoning budget', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

plt.tight_layout()
fig2.savefig('CIFAR10_DarkVFL_ASR.png')

plt.show()