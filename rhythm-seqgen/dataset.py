#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

import numpy as np
import pandas as pd
from fuel.datasets.hdf5 import H5PYDataset
import h5py
from fuel.datasets import IndexableDataset
import json


class T_H5PYDataset(H5PYDataset):
    _encode_dict = {}
    _decode_dict = {}
    _vocab = []
    _vocab_size = 0
    _instances_num = 0

    def __init__(self, file_or_path, which_sets, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        super(T_H5PYDataset, self).__init__(file_or_path, which_sets, subset, load_in_memory, driver, sort_indices, **kwargs)

        with h5py.File(file_or_path) as f:
            self._vocab = json.loads(f.attrs['vocab'])
            self._vocab_size = len(self._vocab)
            self._instances_num = f['inchar'].shape[0]

            self._encode_dict = {word: idx for idx, word in enumerate(self._vocab)}
            self._decode_dict = {idx: word for idx, word in enumerate(self._vocab)}

    def get_data(self, state=None, request=None):
        data = list(super(T_H5PYDataset, self).get_data(state, request))
        data[0] = data[0].T
        data[1] = data[1].T
        return tuple(data)

    def vocab_size(self):
        return self._vocab_size

    def vocab(self):
        return self._vocab

    def encode(self, txt):
        return [self._encode_dict.get(c, 0) for c in txt]

    def decode(self, code):
        return [self._decode_dict.get(i, "<unk>") for i in code]


def createH5Dataset(hdf5_out, corpus_path, sequence_length):
    with open(corpus_path) as f:
        corpus = f.read().split(",")

    (indices, vocab) = pd.factorize(list(corpus))

    instances_num = len(corpus) // (sequence_length + 1)

    f = h5py.File(hdf5_out, mode='w')

    train_data_x = np.zeros((instances_num, sequence_length), dtype=np.uint8)
    train_data_y = np.zeros((instances_num, sequence_length), dtype=np.uint8)

    for j in range(instances_num):
        for i in range(sequence_length):
            train_data_x[j][i] = indices[i + j * (sequence_length + 1)]
            train_data_y[j][i] = indices[i + j * (sequence_length + 1) + 1]

    char_in = f.create_dataset('inchar', train_data_x.shape, dtype='uint8')
    char_out = f.create_dataset('outchar', train_data_y.shape, dtype='uint8')

    char_in[...] = train_data_x
    char_out[...] = train_data_y

    split_dict = {
        'train': {'inchar': (0, instances_num), 'outchar': (0, instances_num)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.attrs["vocab"] = json.dumps(list(vocab))

    f.flush()
    f.close()


class Corpus(object):
    def __init__(self,corpus):
        self.encode_dict = {}
        self.decode_dict = {}
        self.corpus = None
        self.__vocab_size = 0
        self.__init(corpus)

    def __init(self,corpus):
        words,vocab = pd.factorize(corpus.split(","))
        # assign internal vars
        self.corpus = words
        self.__vocab_size = len(vocab)
        self.encode_dict = {word:idx for idx,word in enumerate(vocab)}
        self.decode_dict = {idx:word for idx,word in enumerate(vocab)}

    def get_splits(self,seq_len=70,shifted=False):
        corpus = np.tile(self.corpus, 1)
        corpus = corpus[1:] if shifted else corpus[:-1]
        off_set = len(corpus) % seq_len
        # if last split is unequal the other ones, just skip it
        return corpus[:-off_set].reshape((len(corpus)/seq_len,seq_len)).T

    def vocab_size(self):
        return self.__vocab_size

    def tokenize(self):
        return self.corpus

    def encode(self,txt):
        return [self.encode_dict.get(c, 0) for c in txt]

    def decode(self,code):
        return [self.decode_dict.get(i, "<unk>") for i in code]

if __name__ == "__main__":
    createH5Dataset("dataset/wikifonia-seqlen-100.txt.hdf5", "dataset/wikifonia-seqlen-100.txt", 100)

#train_data = H5PYDataset('dataset.hdf5', which_sets=('train',), load_in_memory=True)
#test_data  = H5PYDataset('dataset.hdf5', which_sets=('test',), load_in_memory=True)
