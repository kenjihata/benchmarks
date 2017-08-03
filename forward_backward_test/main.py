import argparse
import os
from datetime import datetime, timedelta
import torchvision
import torchvision.transforms as transforms
from torchvision.models import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

torch.backends.cudnn.benchmark = True

def millis(start_time):
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='vgg16',
    help='The type of neural network used')
parser.add_argument('--batch_size', type=int, default=16,
    help='The batch size')
parser.add_argument('--num_iterations', type=int, default=1000,
    help='The number of iterations we test on')
parser.add_argument('--cuda', action='store_true',
    help='Use this flag, if you want to use CUDA')
parser.add_argument('--gpu_id', default='0', type=str,
    help='Set id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=1,
    help='Set a random seed')
parser.add_argument('--image_width', type=int, default=224,
    help='The width of the image')
parser.add_argument('--image_height', type=int, default=224,
    help='The height of the image')

args = parser.parse_args()

torch.manual_seed(args.seed)
if args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.cuda.manual_seed(args.seed)

model = resnet.resnet101()
if args.cuda:
    model.cuda()

forward_times = []
backward_times = []

for i in range(args.num_iterations):
    inputs = Variable(torch.rand(args.batch_size, 3, args.image_height, args.image_width))
    if args.cuda:
        inputs = inputs.cuda()
    start_time = datetime.now()
    outputs = model(inputs)
    forward_time = millis(start_time)
    grad = torch.rand(outputs.size()).cuda()
    start_time = datetime.now()
    outputs.backward(gradient=grad)
    backward_time = millis(start_time)
    if i > 0:
        forward_times.append(forward_time)
        backward_times.append(backward_time)

print "Average Forward Time:", np.mean(forward_times), "ms"
print "Average Backward Time:", np.mean(backward_times), "ms"
