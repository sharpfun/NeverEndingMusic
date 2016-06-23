#!/usr/bin/python
#
# author:
#
# date:
# description:
#
import numpy as np
import pandas as pd

import string

import h5py
from fuel.datasets import IndexableDataset

def createH5Dataset(path,corpus):
    f = h5py.File(path, mode='w')
    sequence_length = 25

    in_splits  = corpus.get_splits(seq_len=sequence_length)
    out_splits = corpus.get_splits(seq_len=sequence_length, shifted=True)

    assert in_splits.shape == out_splits.shape

    char_in  = f.create_dataset('inchar',   in_splits.shape, dtype='uint8')
    char_out = f.create_dataset('outchar', out_splits.shape, dtype='uint8')

    char_in[...] = in_splits
    char_out[...] = out_splits

    f.attrs["vocab_size"] = corpus.vocab_size()

    f.flush()
    f.close()

def createDataset(corpus=None, sequence_length=25, repeat=1):
    if not corpus: corpus = Corpus(open("corpus.txt").read())
    vocab_size = corpus.vocab_size()
    in_splits  = corpus.get_splits(seq_len=sequence_length, repeat=repeat)
    out_splits = corpus.get_splits(seq_len=sequence_length, repeat=repeat, shifted=True)

    df = IndexableDataset({
        'inchar': in_splits.astype(np.uint8),
        'outchar': out_splits.astype(np.uint8)
    })

    return df,vocab_size

class Corpus(object):
    def __init__(self,corpus):
        self.encode_dict = {}
        self.decode_dict = {}
        self.corpus = None
        self.__vocab_size = 0
        self.__init(corpus)

    def __init(self,corpus):
        # prepare corpus by replacing useless whitespace
        corpus = corpus.replace(r"  ", " ")
        corpus = [c for c in corpus.lower() if c in string.printable]
        words,vocab = pd.factorize(corpus)
        # assign internal vars
        self.corpus = words
        self.__vocab_size = len(vocab)
        self.encode_dict = {word:idx for idx,word in enumerate(vocab)}
        self.decode_dict = {idx:word for idx,word in enumerate(vocab)}

    def get_splits(self,seq_len=25,repeat=1,shifted=False):
        corpus = np.tile(self.corpus, repeat)
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
    with open("corpus.txt") as fh:
        corpus = Corpus(fh.read())
    words1 = corpus.get_splits()
    words2 = corpus.get_splits(shifted=True)
    print words1.shape
    print words1[:3]
    print words2[:3]
    createH5Dataset("dataset_rnn.hdf5", corpus)

#train_data = H5PYDataset('dataset.hdf5', which_sets=('train',), load_in_memory=True)
#test_data  = H5PYDataset('dataset.hdf5', which_sets=('test',), load_in_memory=True)