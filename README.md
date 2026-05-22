# DarkVFL
the code of DarkVFL
# Requirements
We use a single NVIDIA GeForce RTX 3090 for all experiments. Clone the repository and install the dependencies from requirements.txt using the Anaconda environment:
'''javascript
conda create -n LFBA python=3.9
conda activate LFBA
git clone 'https://github.com/shentt67/LFBA.git'
cd LFBA
pip install requirements.txt
'''
# Example Usage
For instance, to perform backdoor attacks with DarkVFL on the CIFAR10 dataset, run:
'''javascript
python main.py --device 0 --dataset CIFAR10 --epoch 100 --batch_size 256 --lr 0.001 --anchor_idx 23470 --poison_rate 0.1 --select_replace --select_rate 0.5
'''
# Results
