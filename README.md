# TRADES
TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)




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

### Robustness Evaluation:

* Evaluate WideResNet model on CIFAR10 by FGSM-20:
```bash
  -  python pgd_attack_cifar10.py
```

* Evaluate CNN model on MNIST by FGSM-40:
```bash
  -  python pgd_attack_mnist.py
```

 
