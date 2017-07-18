#!/bin/bash

set -e

scp -r herman@server1:/cb/home/herman/imagenet-work/ilsvrc12_train10class_lmdb ..
#exit
#cd bvlc_alexnet10
#caffe train -solver solver.prototxt &> alexnet.log
caffe train -solver solver.prototxt
#cd ..
exit
python extract_alexnet_weights.py
exit
python caffe_surgery.py
