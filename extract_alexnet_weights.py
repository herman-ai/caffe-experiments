import numpy as np
import caffe
import struct
import lmdb

#XXX
net = caffe.Net('/home/ubuntu/bvlc_alexnet10/deploy.prototxt',
        'alexnet-models/bvlc_alexnet10/caffe_alexnet_train10class_nomean_iter_0.caffemodel',
        caffe.TEST)

mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
f = open("data/ilsvrc12/imagenet_mean.binaryproto", 'rb')
mean_blobproto_new.ParseFromString(f.read())
mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
f.close()


print(mean_image[:,:,:4,:4])

lmdb_env = lmdb.open("ilsvrc12_train10class_lmdb")
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

print("input shape = {}".format(net.blobs['data'].data[0].shape))

correct_count = 0
count = 0
for key, value in lmdb_cursor:
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)[:,:227,:227]
    #image[0,:,:] = image[0,:,:] - 109.839
    #image[1,:,:] = image[1,:,:] - 114.914
    #image[2,:,:] = image[2,:,:] - 99.1153
    #image = image.astype(np.uint8)
    #print("image shape = {}".format(image.shape))
    #print(net.blobs['fc8'].data)
    #out = net.forward(data=np.asarray([image])-mean_image[:,:,:227,:227])
    out = net.forward(data=np.asarray([image]))
    print("fc8 : {}".format(net.blobs['fc8'].data))
    print("probabilities : {}".format(net.blobs['prob'].data))
    plabel = int(out['prob'][0].argmax(axis=0))
    print("{} {}".format(plabel, label))
    if plabel==label:
        correct_count += 1
    count += 1
    print("total images seen = {}, total correct = {}".format(count, correct_count))
    break

print("Total correct = {}".format(correct_count))
for key in net.blobs.keys():
    print("{} output shape = {}".format(key, net.blobs[key].data.shape))


for key in net.params.keys():
    shape = net.params[key][0].data.shape 
    print("{} shape = {}".format(key, shape))
    fout = open("propeller_weights/" + key + ".bin", "wb")
    if "conv" in key:
        for r1 in range(shape[2]):
            for r2 in range(shape[3]):
                for c in range(shape[1]):
                    for f in range(shape[0]):
                        fout.write(struct.pack('f', net.params[key][0].data[f, c, r2, r1]))
    elif "fc6" in key:    #XXX
        print("$$here$$")
        for i in range(6):
            for j in range(6):
                for c in range(256):
                    for f in range(4096):
                        fout.write(struct.pack('f', net.params[key][0].data[f, i+j*6+c*36]))
    else:
        for c in range(shape[1]):
            for f in range(shape[0]):
                fout.write(struct.pack('f', net.params[key][0].data[f, c]))
    
    fout.close()

