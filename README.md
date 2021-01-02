# TRADES (**TR**adeoff-inspired **A**dversarial **DE**fense via **S**urrogate-loss minimization) 

This is the code for the [ICML'19 paper](https://arxiv.org/pdf/1901.08573.pdf) "Theoretically Principled Trade-off between Robustness and Accuracy" by [Hongyang Zhang](http://www.cs.cmu.edu/~hongyanz/) (CMU, TTIC), [Yaodong Yu](https://github.com/yaodongyu) (University of Virginia), Jiantao Jiao (UC Berkeley), Eric P. Xing (CMU & Petuum Inc.), Laurent El Ghaoui (UC Berkeley), and Michael I. Jordan (UC Berkeley).

The methodology is the first-place winner of the [NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track)](https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge-robust-model-track/leaderboards).

The attack method transferred from TRADES robust model is the first-place winner of the [NeurIPS 2018 Adversarial Vision Challenge (Targeted Attack Track)](https://www.crowdai.org/challenges/nips-2018-adversarial-vision-challenge-targeted-attack-track/leaderboards).

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
![](http://latex.codecogs.com/gif.latex?\min_f\mathbb{E}\left\\{\mathcal{L}(f(X),Y)+\beta\max_{X'\in\mathbb{B}(X,\epsilon)}\mathcal{L}(f(X),f(X'))\right\\}.)

**Important: the surrogate loss L(.,.) in the second term should be classification-calibrated according to our theory, in contrast to the L2 loss used in [Adversarial Logit Pairing](https://arxiv.org/pdf/1803.06373.pdf).**

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
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
```
### Adversarial training by TRADES:
To apply TRADES, cd into the directory, put 'trades.py' to the directory. Replace ```F.cross_entropy()``` above with ```trades_loss()```:
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
                           beta=args.beta,
			   distance='l_inf')
        loss.backward()
        optimizer.step()
```
#### Arguments:
* ```step_size```: step size for perturbation
* ```epsilon```: limit on the perturbation size
* ```num_steps```: number of perturbation iterations for projected gradient descent (PGD)
* ```beta```: trade-off regularization parameter
* ```distance```: type of perturbation distance, ```'l_inf'``` or ```'l_2'```

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
TRADES won the 1st place out of 1,995 submissions in the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track) on the Tiny ImageNet dataset, surpassing the runner-up approach by 11.41% in terms of L2 perturbation distance.
<p align="center">
    <img src="images/NeurIPS.png" width="450"\>
</p>
<p align="center">
Top-6 results (out of 1,995 submissions) in the NeurIPS 2018 Adversarial Vision Challenge (Robust Model Track). The vertical axis represents the mean L2 perturbation distance that makes robust models fail to output correct labels.
</p>

  
### Certified robustness [[code]](https://github.com/hongyanz/TRADES-smoothing)
TRADES + Random Smoothing achieves SOTA **certified** robustness in ![](http://latex.codecogs.com/gif.latex?\ell_\infty) norm at radius 2/255.
* Results on certified ![](http://latex.codecogs.com/gif.latex?\ell_\infty) robustness at radius 2/255 on CIFAR-10:

| Method              	| Robust Accuracy  	| Natural Accuracy |
|-----------------------|-----------------------|------------------|
| TRADES + Random Smoothing   		|  62.6%   	|   78.7%    		|
| [Salman et al. (2019)](https://arxiv.org/pdf/1906.04584.pdf)   		|  60.8%   	|   82.1%    		|
| [Zhang et al. (2020)](https://arxiv.org/pdf/1906.06316.pdf)   		|  54.0%   	|   72.0%    		|
| [Wong et al. (2018)](https://arxiv.org/pdf/1805.12514.pdf)   		|  53.9%   	|   68.3%    		|
| [Mirman et al. (2018)](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf)   		|  52.2%   	|   62.0%    		|
| [Gowal et al. (2018)](https://arxiv.org/pdf/1810.12715.pdf)   		|  50.0%   	|   70.2%    		|
| [Xiao et al. (2019)](https://arxiv.org/pdf/1809.03008.pdf)   		|  45.9%   	|   61.1%    		|
  

## Want to attack TRADES? No problem!

TRADES is a new baseline method for adversarial defenses. We welcome various attack methods to attack our defense models. We provide checkpoints of our robust models on MNIST dataset and CIFAR dataset. On both datasets, we normalize all the images to ```[0, 1]```.

### How to download our CNN checkpoint for MNIST and WRN-34-10 checkpoint for CIFAR10?
```bash
cd TRADES
mkdir checkpoints
cd checkpoints
```
Then download our pre-trained model

[[download link]](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view?usp=sharing) (CIFAR10)

[[download link]](https://drive.google.com/file/d/1scTd9-YO3-5Ul3q5SJuRrTNX__LYLD_M/view?usp=sharing) (MNIST)

and put them into the folder "checkpoints".

### How to download MNIST dataset and CIFAR10 dataset?
```bash
cd TRADES
mkdir data_attack
cd data_attack
```

Then download the MNIST and CIFAR10 datasets

[[download link]](https://drive.google.com/file/d/1PXePa721gTvmQ46bZogqNGkW31Vu6u3J/view?usp=sharing) (CIFAR10_X)

[[download link]](https://drive.google.com/file/d/1znICoQ8Ds9MH-1yhNssDs3hgBpvx57PV/view?usp=sharing) (CIFAR10_Y)

[[download link]](https://drive.google.com/file/d/12aWmoNs3EMwYe_Z5pBidx_22xj-5IqDU/view?usp=sharing) (MNIST_X)

[[download link]](https://drive.google.com/file/d/1kCBlNfg2TRn8BlqCkNTJiPDgsxIliQgZ/view?usp=sharing) (MNIST_Y)

and put them into the folder "data_attack".



### About the datasets

All the images in both datasets are normalized to ```[0, 1]```.

* ```cifar10_X.npy``` 	-- a ```(10,000, 32, 32, 3)``` numpy array
* ```cifar10_Y.npy``` 	-- a ```(10,000, )``` numpy array
* ```mnist_X.npy``` 	-- a ```(10,000, 28, 28)``` numpy array
* ```mnist_Y.npy``` 	-- a ```(10,000, )``` numpy array

### Load our CNN model for MNIST
```python
from models.small_cnn import SmallCNN

device = torch.device("cuda")
model = SmallCNN().to(device)
model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))
```
For our model ```model_mnist_smallcnn.pt```, the limit on the perturbation size is ```epsilon=0.3``` (L_infinity perturbation distance).

#### White-box leaderboard
| Attack              	| Submitted by  	| Natural Accuracy | Robust Accuracy | Time |
|-----------------------|-----------------------|------------------|-----------------|-----------------|
| [EWR-PGD](https://github.com/liuye6666/EWR-PGD)  	|  Ye Liu (second entry) 	|   99.48%  		|    92.47%   	| Dec 20, 2020
| [EWR-PGD](https://github.com/liuye6666/EWR-PGD)  	|  Ye Liu  	|   99.48%  		|    92.52%    	| Sep 9, 2020
|[Square Attack](https://arxiv.org/abs/1912.00049)		| Andriushchenko Maksym	|   99.48%		|     92.58%	    | Mar 10, 2020
| [fab-attack](https://github.com/fra31/fab-attack)   		|  Francesco Croce   	|   99.48%    		|     93.33%    	| Jun 7, 2019
| FGSM-1,000   		|  (initial entry)  	|     99.48%       |     95.60%      | -
| FGSM-40   		|  (initial entry)   	|     99.48%       |     96.07%      | -

#### How to attack our CNN model on MNIST?
* Step 1: Download ```mnist_X.npy``` and ```mnist_Y.npy```.
* Step 2: Run your own attack on ```mnist_X.npy``` and save your adversarial images as ```mnist_X_adv.npy```.
* Step 3: put ```mnist_X_adv.npy``` under ```./data_attack```.
* Step 4: run the evaluation code,
```bash
  $ python evaluate_attack_mnist.py
```
Note that the adversarial images should in ```[0, 1]``` and the largest perturbation distance is ```epsilon = 0.3```(L_infinity).



### Load our WideResNet (WRN-34-10) model for CIFAR10
```python
from models.wideresnet import WideResNet

device = torch.device("cuda")
model = WideResNet().to(device)
model.load_state_dict(torch.load('./checkpoints/model_cifar_wrn.pt'))
```
For our model ```model_cifar_wrn.pt```, the limit on the perturbation size is ```epsilon=0.031``` (L_infinity perturbation distance).

#### White-box leaderboard

| Attack               	| Submitted by  	| Natural Accuracy 	| Robust Accuracy  	| Time	|
|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| [EWR-PGD](https://github.com/liuye6666/EWR-PGD)  	|  Ye Liu (second entry)   	|   84.92%    		|    52.92%    	| Dec 20, 2020
| [CAA](https://arxiv.org/abs/2012.05434)  	|  Xiaofeng Mao  	|   84.92%    		|    52.94%    	| Dec 14, 2020
| [EWR-PGD](https://github.com/liuye6666/EWR-PGD)  	|  Ye Liu  	|   84.92%    		|    52.95%    	| Sep 9, 2020
| [ODI-PGD](https://arxiv.org/abs/2003.06878)  	|  Yusuke Tashiro  	|   84.92%    		|     53.01%    	| Feb 16, 2020
| [MultiTargeted](https://arxiv.org/abs/1910.09338)   	|  Sven Gowal   	|   84.92%    		|     53.07%    	| Oct 31, 2019
| [AutoAttack](https://github.com/fra31/auto-attack)   	|  (initial entry)   	|   84.92%    		|     53.08%    	| -
| [fab-attack](https://github.com/fra31/fab-attack)   		|  Francesco Croce   	|   84.92%    		|     53.44%    	| Jun 7, 2019
| FGSM-1,000   		|  (initial entry)   	|   84.92%    		|     56.43%    	| -
| FGSM-20   		|  (initial entry)   	|   84.92%    		|     56.61%    	| -
| MI-FGSM	   	|  (initial entry)   	|   84.92%    		|     57.95%    	| -
| FGSM 		   	|  (initial entry)   	|   84.92%    		|     61.06%    	| -
| DeepFool (L_inf)   	|  (initial entry)   	|   84.92%    		|     61.38%    	| -
| CW 		   	|  (initial entry)   	|   84.92%    		|     81.24%    	| -
| DeepFool (L_2)   	|  (initial entry)   	|   84.92%    		|     81.55%    	| -
| LBFGSAttack   	|  (initial entry)   	|   84.92%    		|     81.58%    	| -

#### How to attack our WRM-34-10 model on CIFAR10?
* Step 1: Download ```cifar10_X.npy``` and ```cifar10_Y.npy```.
* Step 2: Run your own attack on ```cifar10_X.npy``` and save your adversarial images as ```cifar10_X_adv.npy```.
* Step 3: put ```cifar10_X_adv.npy``` under ```./data_attack```.
* Step 4: run the evaluation code,
```bash
  $ python evaluate_attack_cifar10.py
```
Note that the adversarial images should be in ```[0, 1]``` and the largest perturbation distance is ```epsilon = 0.031```(L_infinity).


## Reference
For technical details and full experimental results, please check [the paper](https://arxiv.org/pdf/1901.08573.pdf).
```
@inproceedings{zhang2019theoretically, 
	author = {Hongyang Zhang and Yaodong Yu and Jiantao Jiao and Eric P. Xing and Laurent El Ghaoui and Michael I. Jordan}, 
	title = {Theoretically Principled Trade-off between Robustness and Accuracy}, 
	booktitle = {International Conference on Machine Learning},
	year = {2019}
}
```

## Contact
Please contact yyu@eecs.berkeley.edu and hongyanz@ttic.edu if you have any question on the codes. Enjoy!
