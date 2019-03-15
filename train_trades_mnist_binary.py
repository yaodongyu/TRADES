from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.net_mnist import *
from trades import *

parser = argparse.ArgumentParser(description='PyTorch MNIST TRADES Adversarial Training (Binary)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.1,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.01,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='regularization, i.e., lambda in TRADES for binary case')
parser.add_argument('--weight-decay', '--wd', default=0.0,
                    type=float, metavar='W', help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-mnist-net-two-class',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency (default: 10)')
args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# download MNIST dataset
dataset_train = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))

dataset_test = datasets.MNIST('../data', train=False,
                              transform=transforms.Compose([transforms.ToTensor()]))


# select class '1' and class '3'
def get_same_index(target, label_1, label_2):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label_1:
            label_indices.append(i)
        if target[i] == label_2:
            label_indices.append(i)
    return label_indices


# choose 2 classes - '1', '3'
idx_train = get_same_index(dataset_train.train_labels, 1, 3)
dataset_train.train_labels = dataset_train.train_labels[idx_train] - 2
dataset_train.train_data = dataset_train.train_data[idx_train]

# choose 2 classes - '1', '3'
idx_test = get_same_index(dataset_test.test_labels, 1, 3)
dataset_test.test_labels = dataset_test.test_labels[idx_test] - 2
dataset_test.test_data = dataset_test.test_data[idx_test]

# set up dataloader
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=True, **kwargs)


def perturb_hinge(net, x_nat):
    # Perturb function based on (E[\phi(f(x)f(x'))])
    # init with random noise
    net.eval()
    x = x_nat.detach() + 0.001 * torch.randn(x_nat.shape).cuda().detach()
    for _ in range(args.num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            # perturb based on hinge loss
            loss = torch.mean(torch.clamp(1 - net(x).squeeze(1) * (net(x_nat).squeeze(1) / args.beta), min=0))
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + args.step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_nat - args.epsilon), x_nat + args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    net.train()
    return x


def perturb_logistic(net, x_nat, target):
    # Perturb function based on logistic loss
    # init with random noise
    net.eval()
    x = x_nat.detach() + 0.001 * torch.randn(x_nat.shape).cuda().detach()
    for _ in range(args.num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            # perturb based on logistic loss
            loss = torch.mean(1 + torch.exp(-1.0 * target.float() * net(x).squeeze(1)))
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + args.step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, x_nat - args.epsilon), x_nat + args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)
    net.train()
    return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # perturb input x
        x_adv = perturb_hinge(net=model, x_nat=data)

        # optimize
        optimizer.zero_grad()
        output = model(data)
        loss_natural = torch.mean(torch.clamp(1 - output.squeeze(1) * target.float(), min=0))
        loss_robust = torch.mean(torch.clamp(1 - model(x_adv).squeeze(1) * (model(data).squeeze(1) / args.beta), min=0))
        loss = loss_natural + loss_robust
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    """
    evaluate model on training data
    """
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += torch.sum(torch.clamp(1 - target.float() * output.squeeze(1), min=0))
            pred = torch.sign(output).long()
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    # print loss and accuracy
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def eval_test(model, device, test_loader):
    """
    evaluate model on test data
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sum(torch.clamp(1 - target.float() * output.squeeze(1), min=0))
            pred = torch.sign(output).long()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def eval_adv_test(model, device, test_loader):
    """
    evaluate model on test (adversarial) data
    """
    model.eval()
    adv_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # use pgd attack on logistic loss
            x_perturb_linf = perturb_logistic(net=model, x_nat=data, target=target)
            output = model(x_perturb_linf)
            # adversarial loss (E[\phi(f(x)f(x'))])
            adv_loss += torch.sum(torch.clamp(1 - model(x_perturb_linf).squeeze(1) * (model(data).squeeze(1) / args.beta), min=0))
            pred = torch.sign(output).long()
            correct += pred.eq(target.view_as(pred)).sum().item()

    adv_loss /= len(test_loader.dataset)
    print('Test: Average Adv loss: {:.6f}, Robust Accuracy: {}/{} ({:.0f}%)'.format(
        adv_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    model = Net_binary().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural and adversarial examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        eval_adv_test(model, device, test_loader)
        print('================================================================')


if __name__ == '__main__':
    main()
