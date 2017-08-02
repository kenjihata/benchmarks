import argparse
import os
import time
import torchvision
import torchvision.transforms as transforms
from torchvision.models import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
    help='The dataset we will use')
parser.add_argument('--net', type=str, default='vgg16',
    help='The type of neural network used')
parser.add_argument('--lr', type=float, default=0.001,
    help='The learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
    help='Momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
    help='Weight decay')
parser.add_argument('--batch_size', type=int, default=128,
    help='The batch size')
parser.add_argument('--cifar_path', type=str, default='data',
    help='The path to the CIFAR10 dataset')
parser.add_argument('--num_workers', type=int, default=0,
    help='The number of workers to load the data')
parser.add_argument('--num_epochs', type=int, default=10,
    help='The number of epochs we train for')
parser.add_argument('--cuda', action='store_true',
    help='Use this flag, if you want to use CUDA')
parser.add_argument('--gpu_id', default='0', type=str,
    help='Set id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1,
    help='Set a random seed')
parser.add_argument('--batch_print', type=int, default=10,
    help='Prints a sanity check message every x batches')

args = parser.parse_args()

torch.manual_seed(args.seed)
if args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.cuda.manual_seed(args.seed)

model = vgg.vgg16()
model.classifier = nn.Linear(512, 10)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root=args.cifar_path, train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
test_set = torchvision.datasets.CIFAR10(root=args.cifar_path, train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training
def train(epoch):
    model.train()
    train_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % args.batch_print == 0:
            if batch_idx != 0:
                train_loss /= args.batch_print
            print 'Epoch %d, Batch %d/%d | Loss: %f | Accuracy: %f' %\
                  (epoch, batch_idx, len(train_loader), train_loss, correct/total)
            train_loss = 0.
            total = 0.
            correct = 0.

def test(epoch):
    model.eval()
    test_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print 'Epoch %d | Loss: %f | Accuracy: %f' %\
          (epoch, test_loss/len(test_loader), correct/total)

start_time = time.time()

for epoch in range(args.num_epochs):
    train(epoch)
    test(epoch)

time_elapsed = time.time() - start_time
print "Time Elapsed:", time.strftime("%H hours, %M minutes,%S seconds", time.gmtime(time_elapsed))
