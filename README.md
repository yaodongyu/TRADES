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

## Prerequisites:
* Python (3.6.4)
* Pytorch (0.4.1)
* CUDA
* numpy


## TRADES: A New Loss Function for Adversarial Training

### What is TRADES?
TRADES minimizes a regularized surrogate loss L(.,.) (e.g., the cross-entropy loss) for adversarial training:
![](http://latex.codecogs.com/gif.latex?\min_f\mathbb{E}\left\\{\mathcal{L}(f(X),Y)+\max_{X'\in\mathbb{B}(X,\epsilon)}\mathcal{L}(f(X),f(X'))/\lambda\right\\})

The first term encourages the natural error to be optimized by minimizing the "difference" between f(X) and Y , while the second regularization term encourages the output to be smooth, that is, it pushes the decision boundary of classifier away from the sample instances via minimizing the "difference" between the prediction of natural example f(X) and that of adversarial example f(X′). The tuning parameter λ plays a critical role on balancing the importance of natural and robust errors.

<p align="center">
    <img src="images/grid.png" width="450"\>
</p>
Left figure: decision boundary by natural training. Right figure: decision boundary by TRADES.





## How to apply our new loss, TRADES, to train robust models?

### Natural training:
```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
```
### Adversarial training by TRADES:
To apply TRADES, cd into the directory, put file 'trades.py' to the directory. Just need to modify the above code as follows,
```python
from trades import perturb_kl, trades_loss

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        x_adv = perturb_kl(model=model, 
                           x_natural=data,
                           step_size=args.step_size, 
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps)
        optimizer.zero_grad()
        loss = trades_loss(model=model, 
                           x_natural=data,
                           x_adv=x_adv, 
                           y=target,
                           batch_size=args.batch_size, 
                           beta=args.beta)
        loss.backward()
        optimizer.step()
```
#### Arguments:
* --step_size: step size for perturbation
* --epsilon: limit on the perturbation size
* --num_steps: number of perturbation iterations for projected gradient descent (PGD)
* --batch_size: batch size for training
* --beta: trade-off regularization parameter, beta = 1/lambda.

The trade-off regularization parameter ```beta``` can be set in ```[1, 10]```. Larger ```beta``` leads to more robust and less accurate models.

## Running demos

### Adversarial training:

* Train WideResNet-34-10 model on CIFAR10:
```bash
  $ python train_trades_cifar10.py
```

* Train CNN model (four convolutional layers + three fully-connected layers) on MNIST:
```bash
  $ python train_trades_mnist.py
```

* Train CNN model (two convolutional layers + two fully-connected layers) on MNIST (digits '1' and '3') for binary classification problem:
```bash
  $ python train_trades_mnist_binary.py
```

### Robustness evaluation:

* Evaluate robust WideResNet-34-10 model on CIFAR10 by FGSM-20 attack:
```bash
  $ python pgd_attack_cifar10.py
```

* Evaluate robust CNN model on MNIST by FGSM-40 attack:
```bash
  $ python pgd_attack_mnist.py
```


## Experimental results
### Results in the NeurIPS 2018 Adversarial Vision Challenge [[link]](https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge-robust-model-track/leaderboards)
TRADES won the 1st place out of 1,995 submissions in the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track), surpassing the runner-up approach by 11.41% in terms of mean L2 perturbation distance.
<p align="center">
    <img src="images/NeurIPS.png" width="450"\>
</p>


### Results in the Unrestricted Adversarial Examples Challenge [[link]](https://github.com/google/unrestricted-adversarial-examples)

In response to the Unrestricted Adversarial Examples Challenge, we implement a variant of TRADES (with extra spatial-transformation-invariant considerations) on the bird-or-bicycle dataset.

All percentages below correspond to the model's accuracy at 80% coverage.

| Defense               | Submitted by  | Clean data | Common corruptions | Spatial grid attack | SPSA attack | Boundary attack |  Submission Date |
| --------------------- | ------------- | ------------| ------------ |--------------- |-------- | ------- | --------------- |
| [Keras ResNet <br>(trained on ImageNet)](examples/undefended_keras_resnet)   |  Google Brain   |    100.0%    |    99.2%    |  92.2%    |     1.6%    |     4.0%     |  Sept 29th, 2018 |
| [Pytorch ResNet <br>(trained on bird-or-bicycle extras)](examples/undefended_pytorch_resnet)  |  Google Brain | 98.8% | 74.6% | 49.5% | 2.5% | 8.0% | Oct 1st, 2018 |
| [Pytorch ResNet50 <br>(trained on bird-or-bicycle extras)](https://github.com/xincoder/google_attack) |TRADES|100.0%|100.0%|99.5%|100.0%|95.0%|Jan 17th, 2019 (EST)|

## Want to attack TRADES? No problem!

TRADES is a new baseline method for adversarial defenses. We welcome various attack methods to attack our defense models. We provide checkpoints of our robust models on MNIST dataset and CIFAR dataset. On both datasets, we normalize all the images to ```[0, 1]```.

### Load our CNN model for MNIST
```python
from models.small_cnn import SmallCNN

device = torch.device("cuda")
model = SmallCNN().to(device)
model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))
```
For our model ```model_mnist_smallcnn.pt```, the limit on the perturbation size is ```epsilon=0.3``` (L_infinity perturbation distance).


### Load our WideResNet (WRN-34-10) model for CIFAR10
```python
from models.wideresnet import WideResNet

device = torch.device("cuda")
model = WideResNet().to(device)
model.load_state_dict(torch.load('./checkpoints/model_cifar_wrn.pt'))
```
For our model ```model_cifar_wrn.pt```, the limit on the perturbation size is ```epsilon=0.031``` (L_infinity perturbation distance).

## Reference
For technical details and full experimental results, see [the paper]().
```
@article{Zhang2019theoretically, 
	author = {Hongyang Zhang and Yaodong Yu and Jiantao Jiao and Eric P. Xing and Laurent El Ghaoui and Michael I. Jordan}, 
	title = {Theoretically Principled Trade-off between Robustness and Accuracy}, 
	journal={},
	year = {2019}
}
```

## Contact
Please contact yy8ms@virginia.edu and hongyanz@cs.cmu.edu if you have any question on the codes.
