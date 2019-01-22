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


## Usage Examples
### Adversarial Training:

* Train WideResNet model on CIFAR10:
```bash
  -  python train_trades_cifar10.py
```

* Train CNN model (two convolutional layers + two fully-connected layers) on MNIST:
```bash
  -  python train_trades_mnist.py
```

* Train CNN model (two convolutional layers + two fully-connected layers) on MNIST (digit `1' and `3') for binary classification:
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

 
