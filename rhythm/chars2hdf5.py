import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu"

import h5py
from fuel.datasets.hdf5 import H5PYDataset
import numpy
import pandas
import json

if __name__ == "__main__":
    corpus_path = 'dataset/finalfile-seqlen-100.txt'
    hdf5_out = 'dataset/finalfile-seqlen-100.hdf5'
    seqlength = 70

    with open(corpus_path) as f:
        corpus = f.read()

    (indices, indexed_letters) = pandas.factorize(list(corpus))

    instances_num = (len(corpus)-1) // (seqlength+1)

    f = h5py.File(hdf5_out, mode='w')

    train_data_x = numpy.zeros((instances_num, seqlength), dtype=numpy.uint8)
    train_data_y = numpy.zeros((instances_num, seqlength), dtype=numpy.uint8)

    for j in range(instances_num):
        for i in range(seqlength):
            train_data_x[j][i] = indices[i + j * (seqlength+1)]
            train_data_y[j][i] = indices[i + j * (seqlength+1) + 1]

    fx = f.create_dataset('x', train_data_x.shape, dtype='uint8')
    fy = f.create_dataset('y', train_data_y.shape, dtype='uint8')

    fx[...] = train_data_x
    fy[...] = train_data_y

    split_dict = {
        'train': {'x': (0, instances_num), 'y': (0, instances_num)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.attrs['index_to_char'] = json.dumps(list(indexed_letters))

    f.flush()
    f.close()
