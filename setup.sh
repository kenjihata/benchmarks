#!/bin/bash

mkdir -p data
cd data

if [ ! -d "cifar-10-batches-py" ] || [ "$1" == "reset" ]; then
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    rm -f cifar-10-python.tar.gz
fi
