# coding: utf-8
import caffe
import struct
mean_blob = caffe.proto.caffe_pb2.BlobProto()
mean_blob.ParseFromString(open("data/ilsvrc12/imagenet_mean.binaryproto", "rb").read())
mean_array = caffe.io.blobproto_to_array(mean_blob)

f = open("mean.bin", "wb")
for h in range(256):
    for w in range(256):
        for c in range(3):
            f.write(struct.pack('f', mean_array[0, c, w, h]))
f.close()
