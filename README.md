# TRADES (**TR**adeoff-inspired **A**dversarial **DE**fense via **S**urrogate-loss minimization) 

This is the code for the paper "Theoretically Principled Trade-off between Robustness and Accuracy" by Hongyang Zhang (CMU), Yaodong Yu (University of Virginia), Jiantao Jiao (UC Berkeley), Eric P. Xing (CMU & Petuum Inc.), Laurent El Ghaoui (UC Berkeley), and Michael I. Jordan (UC Berkeley).

The code is written in python and requires numpy, matplotlib, torch, torchvision and the tqdm library.

## Install
This code depends on python 3.6, pytorch 0.4.1. We suggest to install the dependencies using Anaconda or Miniconda. Here is an exemplary command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install pytorch=0.4.1
```

## Running Demos
### Adversarial Training:

* Train WideResNet-34-10 model on CIFAR10:
```bash
  -  python train_trades_cifar10.py
```

* Train CNN model (two convolutional layers + two fully-connected layers) on MNIST:
```bash
  -  python train_trades_mnist.py
```

* Train CNN model (two convolutional layers + two fully-connected layers) on MNIST (digits '1' and '3') for binary classification:
```bash
  -  python train_trades_mnist_binary.py
```

### Robustness Evaluation:

* Evaluate robust WideResNet-34-10 model on CIFAR10 by FGSM-20 attack:
```bash
  -  python pgd_attack_cifar10.py
```

* Evaluate robust CNN model on MNIST by FGSM-40 attack:
```bash
  -  python pgd_attack_mnist.py
```

## TRADES: A New Loss Function for Adversarial Training

### How to import the TRADES loss for your adversarial training?
* To get started, cd into the directory. Put file 'trades.py' to the directory. Then write the following head in your running 'xxx.py' file and replace your loss with TRADES_loss():
```bash
  -  from trades import TRADES_loss
```

## Experimental Results
### Results in the NeurIPS 2018 Adversarial Vision Challenge
TRADES won the 1st place out of 1,995 submissions in the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track), surpassing the runner-up approach by 11.41% in terms of mean l2 perturbation distance.
<p align="center">
    <img src="NeurIPS.png" width="600"\>
</p>


### Results in the Unrestricted Adversarial Examples Challenge
