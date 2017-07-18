# coding: utf-8
import os
os.environ['GLOG_minloglevel'] = '0'
import caffe
import numpy as np
import lmdb
import sys

solver = caffe.SGDSolver("/home/ubuntu/bvlc_alexnet10/solver.prototxt")

solver.step(1)
fc8w0 = np.copy(solver.net.params["fc8"][0].data)
fc8g0 = np.copy(solver.net.params["fc8"][0].diff)
solver.step(1)
fc8w1 = np.copy(solver.net.params["fc8"][0].data)
fc8g1 = np.copy(solver.net.params["fc8"][0].diff)
sys.exit(0)

image = solver.net.blobs["data"].data

pool1 = solver.net.blobs["pool1"].data
pool2 = solver.net.blobs["pool2"].data
pool5 = solver.net.blobs["pool5"].data

pool1d = solver.net.blobs["pool1"].diff
pool2d = solver.net.blobs["pool2"].diff
pool5d = solver.net.blobs["pool5"].diff

conv1 = solver.net.blobs["conv1"].data
conv2 = solver.net.blobs["conv2"].data
conv3 = solver.net.blobs["conv3"].data
conv4 = solver.net.blobs["conv4"].data
conv5 = solver.net.blobs["conv5"].data

conv1d = solver.net.blobs["conv1"].diff
conv2d = solver.net.blobs["conv2"].diff
conv3d = solver.net.blobs["conv3"].diff
conv4d = solver.net.blobs["conv4"].diff
conv5d = solver.net.blobs["conv5"].diff

fc6 = solver.net.blobs["fc6"].data
fc7 = solver.net.blobs["fc7"].data
fc8 = solver.net.blobs["fc8"].data

fc8d = solver.net.blobs["fc8"].diff
fc7d = solver.net.blobs["fc7"].diff
fc6d = solver.net.blobs["fc6"].diff

for a in zip(["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "fc8"],
            [conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8]):
    np.save("../activations/"+a[0], a[1])

for a in zip(["pool1", "pool2", "pool5"], [pool1, pool2, pool5]):
    np.save("../activations/"+a[0], a[1])

for a in zip(["pool1", "pool2", "pool5"], [pool1d, pool2d, pool5d]):
    np.save("../deltas/"+a[0], a[1])

for a in zip(["conv1", "conv2", "conv3", "conv4", "conv5","fc8", "fc7", "fc6"],
            [conv1d, conv2d, conv3d, conv4d, conv5d, fc8d, fc7d, fc6d]):
    np.save("../deltas/"+a[0], a[1])

