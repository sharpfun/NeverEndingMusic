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
import json


class T_H5PYDataset(H5PYDataset):
    _syllables_encode_dict = {}
    _syllables_decode_dict = {}
    _syllables_vocab = []

    _stress_encode_dict = {}
    _stress_decode_dict = {}
    _stress_vocab = []

    _durations_encode_dict = {}
    _durations_decode_dict = {}
    _durations_vocab = []

    _pitches_encode_dict = {}
    _pitches_decode_dict = {}
    _pitches_vocab = []

    _phrase_encode_dict = {}
    _phrase_decode_dict = {}
    _phrase_vocab = []

    def __init__(self, file_or_path, which_sets, subset=None,
                 load_in_memory=False, driver=None, sort_indices=True,
                 **kwargs):
        super(T_H5PYDataset, self).__init__(file_or_path, which_sets, subset, load_in_memory, driver, sort_indices, **kwargs)

        with h5py.File(file_or_path) as f:
            self._syllables_vocab = json.loads(f.attrs['syllables_vocab'])
            self._syllables_encode_dict = {word: idx for idx, word in enumerate(self._syllables_vocab)}
            self._syllables_decode_dict = {idx: word for idx, word in enumerate(self._syllables_vocab)}

            self._stress_vocab = json.loads(f.attrs['stress_vocab'])
            self._stress_encode_dict = {word: idx for idx, word in enumerate(self._stress_vocab)}
            self._stress_decode_dict = {idx: word for idx, word in enumerate(self._stress_vocab)}

            self._durations_vocab = json.loads(f.attrs['durations_vocab'])
            self._durations_encode_dict = {word: idx for idx, word in enumerate(self._durations_vocab)}
            self._durations_decode_dict = {idx: word for idx, word in enumerate(self._durations_vocab)}

            self._pitches_vocab = json.loads(f.attrs['pitches_vocab'])
            self._pitches_encode_dict = {word: idx for idx, word in enumerate(self._pitches_vocab)}
            self._pitches_decode_dict = {idx: word for idx, word in enumerate(self._pitches_vocab)}

            self._phrase_vocab = json.loads(f.attrs['phrase_vocab'])
            self._phrase_encode_dict = {word: idx for idx, word in enumerate(self._phrase_vocab)}
            self._phrase_decode_dict = {idx: word for idx, word in enumerate(self._phrase_vocab)}

    def get_data(self, state=None, request=None):
        data = list(super(T_H5PYDataset, self).get_data(state, request))
        data[0] = data[0].T
        data[1] = data[1].T
        data[2] = data[2].T
        data[3] = data[3].T
        data[4] = data[4].T
        data[5] = data[5].T
        return tuple(data)

    def syllables_vocab_size(self):
        return len(self._syllables_vocab)

    def syllables_vocab(self):
        return self._syllables_vocab

    def phrase_vocab(self):
        return self._phrase_vocab

    def phrase_vocab_size(self):
        return len(self._phrase_vocab)

    def phrase_idx_encode(self, txt):
        return [self._syllables_encode_dict.get(c, self._phrase_encode_dict.get(0)) for c in txt]

    def phrase_idx_decode(self, code):
        return [self._phrase_decode_dict[i] for i in code]

    def syllables_encode(self, txt):
        return [self._syllables_encode_dict.get(c, self._syllables_encode_dict.get("<unk>")) for c in txt]

    def syllables_decode(self, code):
        return [self._syllables_decode_dict[i] for i in code]

    def stress_vocab_size(self):
        return len(self._stress_vocab)

    def stress_vocab(self):
        return self._stress_vocab

    def stress_encode(self, txt):
        return [self._stress_encode_dict.get(c, self._stress_encode_dict.get("<unk>")) for c in txt]

    def stress_decode(self, code):
        return [self._stress_decode_dict[i] for i in code]

    def durations_vocab_size(self):
        return len(self._durations_vocab)

    def durations_vocab(self):
        return self._durations_vocab

    def durations_encode(self, txt):
        return [self._durations_encode_dict[c] for c in txt]

    def durations_decode(self, code):
        return [self._durations_decode_dict[i] for i in code]

    def pitches_vocab_size(self):
        return len(self._pitches_vocab)

    def phrase_index_vocab_size(self):
        return len(self._)

    def pitches_vocab(self):
        return self._pitches_vocab

    def pitches_encode(self, txt):
        return [self._pitches_encode_dict[c] for c in txt]

    def pitches_decode(self, code):
        return [self._pitches_decode_dict[i] for i in code]


def createH5Dataset(hdf5_out, normalized_outfile, sequence_length):
    with open(normalized_outfile) as f:
        durations_set = set()
        syllables_dict = dict()
        pitches_set = set()

        sheets = json.loads(f.read())
        for sheet in sheets:
            for i in range(len(sheet["durations"])):
                durations_set.add(sheet["durations"][i])

                syllable = sheet["syllables"][i]
                if syllable not in syllables_dict:
                    syllables_dict[syllable] = set()
                syllables_dict[syllable].add(sheet["file_path"])
                pitches_set.add(sheet["pitches"][i])

        syllables_set = set()

        for key, value in syllables_dict.iteritems():
            if len(value) > 10:
                syllables_set.add(key)

        for sheet in sheets:
            for i in range(len(sheet["durations"])):
                if sheet["syllables"][i] not in syllables_set:
                    sheet["syllables"][i] = "<unk>"

        all_durations = []
        all_syllables = []
        all_pitches = []
        all_stress = []
        all_end_idx = []

        """for sheet in sheets:
            upto = sequence_length*(len(sheet["durations"])/sequence_length)
            sheet["durations"] = sheet["durations"][:upto]
            sheet["syllables"] = sheet["syllables"][:upto]
            sheet["pitches"] = sheet["pitches"][:upto]

        for sheet in sheets:
            all_durations += sheet["durations"]
            all_syllables += sheet["syllables"]
            all_pitches += sheet["pitches"]"""

        for sheet in sheets:
            all_durations += sheet["durations"] + [0]
            all_syllables += sheet["syllables"] + ["<end_file>"]
            all_pitches += sheet["pitches"] + ["R"]
            all_stress += sheet["stress"] + ["_"]
            all_end_idx += sheet["phrase_end"] + [1]

        print durations_set
        print len(durations_set)

        print syllables_set
        print len(syllables_set)

        print pitches_set
        print len(pitches_set)

        (durations_indices, durations_vocab) = pd.factorize(all_durations)
        (syllables_indices, syllables_vocab) = pd.factorize(all_syllables)
        (pitches_indices, pitches_vocab) = pd.factorize(all_pitches)
        (stress_indices, stress_vocab) = pd.factorize(all_stress)

        instances_num = (len(durations_indices)-1) // sequence_length
        fout = h5py.File(hdf5_out, mode='w')

        train_data_durations = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_durations_shift = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_syllables = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_pitches = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_pitches_shift = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_stress = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_phrase_end = np.zeros((instances_num, sequence_length), dtype=np.uint16)

        for j in range(instances_num):
            for i in range(sequence_length):
                train_data_durations[j][i] = durations_indices[i + j * sequence_length]
                train_data_durations_shift[j][i] = durations_indices[i + j * sequence_length + 1]
                train_data_syllables[j][i] = syllables_indices[i + j * sequence_length]
                train_data_pitches[j][i] = pitches_indices[i + j * sequence_length]
                train_data_pitches_shift[j][i] = pitches_indices[i + j * sequence_length + 1]
                train_data_stress[j][i] = stress_indices[i + j * sequence_length]
                train_data_phrase_end[j][i] = all_end_idx[i + j * sequence_length]

        note_durations = fout.create_dataset('durations', train_data_durations.shape, dtype='uint16')
        note_durations_shift = fout.create_dataset('durations_shift', train_data_durations.shape, dtype='uint16')
        note_syllables = fout.create_dataset('syllables', train_data_syllables.shape, dtype='uint16')
        note_pitches = fout.create_dataset('pitches', train_data_syllables.shape, dtype='uint16')
        note_pitches_shift = fout.create_dataset('pitches_shift', train_data_syllables.shape, dtype='uint16')
        note_stress = fout.create_dataset('stress', train_data_stress.shape, dtype='uint16')
        note_end = fout.create_dataset('phrase_end', train_data_phrase_end.shape, dtype='uint16')

        note_durations[...] = train_data_durations
        note_durations_shift[...] = train_data_durations_shift
        note_syllables[...] = train_data_syllables
        note_pitches[...] = train_data_pitches
        note_pitches_shift[...] = train_data_pitches_shift
        note_stress[...] = train_data_stress
        note_end[...] = train_data_phrase_end

        split_num = int(instances_num * 0.7)

        split_dict = {
            'train': {
                'durations': (0, split_num),
                'durations_shift': (0, split_num),
                'syllables': (0, split_num),
                'pitches': (0, split_num),
                'pitches_shift': (0, split_num),
                'stress': (0, split_num),
                'phrase_end': (0, split_num)},
            'test': {
                'durations': (split_num+1, instances_num),
                'durations_shift': (split_num+1, instances_num),
                'syllables': (split_num+1, instances_num),
                'pitches': (split_num+1, instances_num),
                'pitches_shift': (split_num+1, instances_num),
                'stress': (split_num+1, instances_num),
                'phrase_end': (split_num+1, instances_num)}
        }

        fout.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        print [int(x) for x in list(durations_vocab)]
        fout.attrs["durations_vocab"] = json.dumps([int(x) for x in list(durations_vocab)])
        fout.attrs["syllables_vocab"] = json.dumps(list(syllables_vocab))
        fout.attrs["pitches_vocab"] = json.dumps(list(pitches_vocab))
        fout.attrs["stress_vocab"] = json.dumps(list(stress_vocab))
        fout.attrs["phrase_vocab"] = json.dumps([0, 1])

        fout.flush()
        fout.close()


if __name__ == "__main__":
    createH5Dataset('dataset/normalized_syllables_rhythm_notes.json-seqlen-100.hdf5',
                    'dataset/normalized_syllables_rhythm_notes.json', 100)
    ds = T_H5PYDataset('dataset/normalized_syllables_rhythm_notes.json-seqlen-30.hdf5', which_sets=('train',))
    print ds.syllables_vocab()

#train_data = H5PYDataset('dataset.hdf5', which_sets=('train',), load_in_memory=True)
#test_data  = H5PYDataset('dataset.hdf5', which_sets=('test',), load_in_memory=True)
