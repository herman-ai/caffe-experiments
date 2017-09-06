#!/bin/bash

set -e

# Note: LR is the learning rate for a batch size of 256

for var in LR ITERS_PER_EPOCH ITERS_PER_FIFTY_EPOCHS BATCH_SIZE
do
    if [ -z $"${!var}" ]
    then
        echo $var is not set
        exit 1
    fi
done


echo "Running on :"$(hostname --ip-address)
DIR=/spare/jenkins/workspace/caffe-experiments/caffe-experiments
#DIR=$(pwd)

#cp -r /cb/home/herman/ws/caffe-experiments/* $DIR
#cp -r /cb/home/herman/imagenet-work/ilsvrc12_train10class_lmdb $DIR

NORMALIZED_LR=$(python -c "print(${LR}/256*${BATCH_SIZE})")

echo "LR = ${LR}, NORMALIZED_LR = ${NORMALIZED_LR}"

sed "s/LR/${NORMALIZED_LR}/" solver.prototxt.template > ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt
sed -i "s|DIR|${DIR}|" ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt
sed -i "s/ITERS_PER_EPOCH/${ITERS_PER_EPOCH}/" ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt
sed -i "s/ITERS_PER_FIFTY_EPOCHS/${ITERS_PER_FIFTY_EPOCHS}/" ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt

sed "s/BATCH_SIZE/${BATCH_SIZE}/" train_val.prototxt.template > ${DIR}/train_val-${BATCH_SIZE}.prototxt
sed -i "s/BATCH_SIZE/${BATCH_SIZE}/" ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt

echo "caffe train -solver solver-${LR}-${BATCH_SIZE}.prototxt > train_log_${LR}_${BATCH_SIZE}.txt"
/cb/home/herman/caffe/build/tools/caffe train -solver ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt > ${DIR}/train_log_${LR}_${BATCH_SIZE}.txt 2>&1
s3cmd put ${DIR}/train_log_${LR}_${BATCH_SIZE}.txt s3://caffe-experiments/mb${BATCH_SIZE}/train_log_${LR}.txt

rm -f ${DIR}/train_log_${LR}_${BATCH_SIZE}.txt ${DIR}/solver-${LR}-${BATCH_SIZE}.prototxt ${DIR}/*.caffemodel ${DIR}/*.solverstate
rm -rf ${DIR}/ilsvrc12_train10class_lmdb

