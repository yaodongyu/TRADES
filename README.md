# TRADES (**TR**adeoff-inspired **A**dversarial **DE**fense via **S**urrogate-loss minimization) 

This is the code for the paper "Theoretically Principled Trade-off between Robustness and Accuracy" by Hongyang Zhang (CMU), Yaodong Yu (University of Virginia), Jiantao Jiao (UC Berkeley), Eric P. Xing (CMU & Petuum Inc.), Laurent El Ghaoui (UC Berkeley), and Michael I. Jordan (UC Berkeley).

The methodology is the winner of the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track).

## Prerequisites
* Python (3.6.4)
* Pytorch (0.4.1)
* CUDA
* numpy

## Install
We suggest to install the dependencies using Anaconda or Miniconda. Here is an exemplary command:
```
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh
$ source ~/.bashrc
$ conda install pytorch=0.4.1
```


## TRADES: A New Loss Function for Adversarial Training

### What is TRADES?
TRADES minimizes a regularized surrogate loss L(.,.) (e.g., the cross-entropy loss) for adversarial training:
![](http://latex.codecogs.com/gif.latex?\min_f\mathbb{E}\left\\{\mathcal{L}(f(X),Y)+\beta\max_{X'\in\mathbb{B}(X,\epsilon)}\mathcal{L}(f(X),f(X'))\right\\})

The first term encourages the natural error to be optimized by minimizing the "difference" between f(X) and Y , while the second regularization term encourages the output to be smooth, that is, it pushes the decision boundary of classifier away from the sample instances via minimizing the "difference" between the prediction of natural example f(X) and that of adversarial example f(X′). The tuning parameter β plays a critical role on balancing the importance of natural and robust errors.

<p align="center">
    <img src="images/grid.png" width="450"\>
</p>
<p align="center">
<b>Left figure:</b> decision boundary by natural training. <b>Right figure:</b> decision boundary by TRADES.
</p>




## How to use TRADES to train robust models?

### Natural training:
```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
```
### Adversarial training by TRADES:
To apply TRADES, cd into the directory, put 'trades.py' to the directory. Replace ```F.nll_loss()``` above with ```trades_loss()```:
```python
from trades import trades_loss

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # calculate robust loss - TRADES loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           batch_size=args.batch_size,
                           beta=args.beta)
        loss.backward()
        optimizer.step()
```
#### Arguments:
* ```step_size```: step size for perturbation
* ```epsilon```: limit on the perturbation size
* ```num_steps```: number of perturbation iterations for projected gradient descent (PGD)
* ```batch_size```: batch size for training
* ```beta```: trade-off regularization parameter.

The trade-off regularization parameter ```beta``` can be set in ```[1, 10]```. Larger ```beta``` leads to more robust and less accurate models.

### Basic MNIST example (adversarial training by TRADES):
```python
python mnist_example_trades.py
```
We adapt ```main.py``` in [[link]](https://github.com/pytorch/examples/tree/master/mnist) to our new loss ```trades_loss()``` during training.



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
<p align="center">
Top-6 results (out of 1,995 submissions) in the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track). The vertical axis represents the mean l2 perturbation distance that makes robust models fail to output correct labels.
</p>

### Results in the Unrestricted Adversarial Examples Challenge [[link]](https://github.com/google/unrestricted-adversarial-examples)

In response to the Unrestricted Adversarial Examples Challenge, we implement a variant of TRADES (with extra spatial-transformation-invariant considerations) on the bird-or-bicycle dataset.

All percentages below correspond to the model's accuracy at 80% coverage.

| Defense               | Submitted by  | Clean data | Common corruptions | Spatial grid attack | SPSA attack | Boundary attack |  Submission Date |
| --------------------- | ------------- | ------------| ------------ |--------------- |-------- | ------- | --------------- |
| [Keras ResNet <br>(trained on ImageNet)](https://github.com/google/unrestricted-adversarial-examples/tree/master/examples/undefended_keras_resnet)   |  Google Brain   |    100.0%    |    99.2%    |  92.2%    |     1.6%    |     4.0%     |  Sept 29th, 2018 |
| [Pytorch ResNet <br>(trained on bird-or-bicycle extras)](https://github.com/google/unrestricted-adversarial-examples/tree/master/examples/undefended_pytorch_resnet)  |  Google Brain | 98.8% | 74.6% | 49.5% | 2.5% | 8.0% | Oct 1st, 2018 |
| [Pytorch ResNet50 <br>(trained on bird-or-bicycle extras)](https://github.com/xincoder/google_attack) |TRADES|100.0%|100.0%|99.5%|100.0%|95.0%|Jan 17th, 2019 (EST)|

## Want to attack TRADES? No problem!

TRADES is a new baseline method for adversarial defenses. We welcome various attack methods to attack our defense models. We provide checkpoints of our robust models on MNIST dataset and CIFAR dataset. On both datasets, we normalize all the images to ```[0, 1]```.

### How to download our CNN checkpoint for MNIST and WRN-34-10 checkpoint for CIFAR10?
```bash
cd TRADES
mkdir checkpoints
cd checkpoints
wget http://people.virginia.edu/~yy8ms/TRADES/model_mnist_smallcnn.pt
wget http://people.virginia.edu/~yy8ms/TRADES/model_cifar_wrn.pt
```

### How to download MNIST dataset and CIFAR10 dataset?
```bash
cd TRADES
mkdir data_attack
cd data_attack
wget http://people.virginia.edu/~yy8ms/TRADES/cifar10_X.npy
wget http://people.virginia.edu/~yy8ms/TRADES/cifar10_Y.npy
wget http://people.virginia.edu/~yy8ms/TRADES/mnist_X.npy
wget http://people.virginia.edu/~yy8ms/TRADES/mnist_Y.npy
```

### How to download MNIST dataset and CIFAR10 dataset?
```bash
cd TRADES
mkdir data_attack
cd data_attack
wget http://people.virginia.edu/~yy8ms/TRADES/cifar10_X.npy
wget http://people.virginia.edu/~yy8ms/TRADES/cifar10_Y.npy
wget http://people.virginia.edu/~yy8ms/TRADES/mnist_X.npy
wget http://people.virginia.edu/~yy8ms/TRADES/mnist_Y.npy
```

### About the datasets

```cifar10_X.npy``` 	-- a ```(10000, 32, 32, 3)``` numpy array
```cifar10_Y.npy``` 	-- a ```(10000, )``` numpy array
```mnist_X.npy``` 	-- a ```(10000, 28, 28)``` numpy array
```mnist_Y.npy``` 	-- a ```(10000, )``` numpy array

### Load our CNN model for MNIST
```python
from models.small_cnn import SmallCNN

device = torch.device("cuda")
model = SmallCNN().to(device)
model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))
```
For our model ```model_mnist_smallcnn.pt```, the limit on the perturbation size is ```epsilon=0.3``` (L_infinity perturbation distance).

#### White-box leaderboard
| Attack               | Submitted by  | Natural Accuracy | Robust Accuracy |
| --------------------- | ------------- | ------------| ------------ |
| FGSM-40   |  (initial entry)   |     99.48%    |     96.07%    |


### Load our WideResNet (WRN-34-10) model for CIFAR10
```python
from models.wideresnet import WideResNet

device = torch.device("cuda")
model = WideResNet().to(device)
model.load_state_dict(torch.load('./checkpoints/model_cifar_wrn.pt'))
```
For our model ```model_cifar_wrn.pt```, the limit on the perturbation size is ```epsilon=0.031``` (L_infinity perturbation distance).

#### White-box leaderboard

| Attack               	| Submitted by  	| Natural Accuracy 	| Robust Accuracy  	|
|-----------------------|-----------------------|-----------------------|-----------------------|
| FGSM-20   		|  (initial entry)   	|   84.92%    		|     56.61%    	|
| DeepFool (L_inf)   	|  (initial entry)   	|   84.92%    		|     61.38%    	|
| DeepFool (L_2)   	|  (initial entry)   	|   84.92%    		|     81.55%    	|
| LBFGSAttack   	|  (initial entry)   	|   84.92%    		|     81.58%    	|
| MI-FGSM	   	|  (initial entry)   	|   84.92%    		|     57.95%    	|
| CW 		   	|  (initial entry)   	|   84.92%    		|     81.24%    	|
| FGSM 		   	|  (initial entry)   	|   84.92%    		|     61.06%    	|

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
