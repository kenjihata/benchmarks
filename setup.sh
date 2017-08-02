#!/bin/bash

mkdir -p data
cd data

if [ ! -d "annotations" ] || [ "$1" == "reset" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    mv cifar-10-python cifar10
fi
