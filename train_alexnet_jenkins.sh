#!/bin/bash

set -e

echo "Running on :"$(hostname)
DIR=/spare/jenkins/workspace/caffe-experiments/caffe-experiments
#DIR=$(pwd)

cp -r /cb/home/herman/ws/caffe-experiments/* $DIR
cp -r /cb/home/herman/imagenet-work/ilsvrc12_train10class_lmdb $DIR

sed "s/LR/${LR}/" solver.prototxt.template > ${DIR}/solver-${LR}.prototxt
sed -i "s|DIR|${DIR}|" ${DIR}/solver-${LR}.prototxt

echo "caffe train -solver solver-${LR}.prototxt > train_log_${LR}.txt"
/cb/home/herman/caffe/build/tools/caffe train -solver ${DIR}/solver-${LR}.prototxt > ${DIR}/train_log_${LR}.txt 2>&1
s3cmd put ${DIR}/train_log_${LR}.txt s3://caffe-experiments/mb256/train_log_${LR}.txt
rm -f ${DIR}/train_log_${LR}.txt
#TODO: Also remove the checkpoint and solver.prototxt

