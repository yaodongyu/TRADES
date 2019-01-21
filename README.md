# TRADES
**TRADES** (**TR**adeoff-inspired **A**dversarial **DE**fense via **S**urrogate-loss minimization)




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

 
