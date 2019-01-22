# TRADES (**TR**adeoff-inspired **A**dversarial **DE**fense via **S**urrogate-loss minimization) 

This is the code for the paper "Theoretically Principled Trade-off between Robustness and Accuracy".

The code is written in python and requires numpy, matplotlib, torch, torchvision and the tqdm library.

## Install
This code depends on python 3.6, pytorch 0.4.1. We suggest to install the dependencies using Anaconda or Miniconda. Here is an exemplary command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install pytorch=0.4.1
```

## Get started
To get started, cd into the directory. Then runs the scripts: 
* gan_stackelberg_mG.py is a demo on the performance of Stackelberg GAN on Gaussian mixture dataset,
* gan_branch_mG.py is a demo on the performance of multi-branch GAN (a baseline method) on Gaussian mixture dataset,
* gan_mnist_classifier.py is a demo on the performance of Stackelberg GAN on MNIST dataset,
* gan_mnist_fashion_classifier.py is a demo on the performance of Stackelberg GAN on fashion-MNIST dataset.

## Using the code
The command `python xxx.py --help` gives the help information about how to run the code.

## Usage Examples:
### Adversarial Training:

* Train WideResNet model on CIFAR10:
```bash
  -  python train_trades_cifar10.py
```

* Train CNN model on MNIST:
```bash
  -  python train_trades_mnist.py
```

* Train CNN model on MNIST for binary classification:
```bash
  -  python train_trades_mnist_binary.py
```

### Robustness Evaluation:

* Evaluate WideResNet model on CIFAR10 by FGSM-20:
```bash
  -  python pgd_attack_cifar10.py
```

* Evaluate CNN model on MNIST by FGSM-40:
```bash
  -  python pgd_attack_mnist.py
```

 
