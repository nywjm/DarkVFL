# ShadowVFL
the code of ShadowVFL
# Requirements
We use a single NVIDIA GeForce RTX 3090 for all experiments. Clone the repository and install the dependencies from requirements.txt using the Anaconda environment:
```bash
conda create -n ShadowVFL python=3.9
conda activate ShadowVFL
git clone 'https://github.com/nywjm/ShadowVFL.git'
cd ShadowVFL
pip install requirements.txt
```
# Example Usage
For instance, to perform backdoor attacks with ShadowVFL on the CIFAR10 dataset, run:
```bash
python main.py --device 0 --dataset CIFAR10 --epoch 100 --batch_size 256 --lr 0.001 --anchor_idx 23470 --poison_rate 0.1 --select_replace --select_rate 0.5
```
# Results
## ASR
|                                                    **Method**                                                   |       **MNIST**      | **FashionMNIST**                                         | **CIFAR-10**                                                    |
|:---------------------------------------------------------------------------------------------------------------:|:---------------------:|------------------------------------------------------------|-------------------------------------------------------------|
|             ShadowVFL             |   88.81   | 79.18                                               | 99.77                                      |
|             LFBA            |   64.88   | 68.68                                               | 97.30                                     |
|             BadVFL             |   63.43   | 63.40                                               | 85.00                                      |
|             Vanilla             |   1.57   | 5.51                                              | 6.96                                     |

