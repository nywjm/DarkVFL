# DarkVFL
the code of DarkVFL
# Requirements
We use a single NVIDIA GeForce RTX 3090 for all evaluations. Clone the repository and install the dependencies from requirements.txt using the Anaconda environment:
```bash
conda create -n DarkVFL python=3.9
conda activate DarkVFL
git clone 'https://github.com/nywjm/DarkVFL.git'
cd DarkVFL
pip install requirements.txt
```
# Example Usage
For instance, to perform backdoor attacks with DarkVFL on the CIFAR10 dataset, run:
```bash
python main.py --device 0 --dataset CIFAR10 --epoch 100 --batch_size 256 --lr 0.001 --anchor_idx 23470 --poison_rate 0.1 --select_replace --select_rate 0.5
```
# Results
