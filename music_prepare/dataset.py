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
        for i in range(len(data)):
            data[i] = data[i].T
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
        all_durations_prev = []
        all_syllables = []
        all_pitches = []
        all_pitches_prev = []
        all_stress = []
        all_end_idx = []

        """
        for sheet in sheets:
            upto = sequence_length*((len(sheet["durations"])-1)/sequence_length)
            all_durations += sheet["durations"][1:upto+1]
            all_durations_prev += sheet["durations"][:upto]
            all_syllables += sheet["syllables"][:upto]
            all_pitches += sheet["pitches"][1:upto+1]
            all_pitches_prev += sheet["pitches"][:upto]
            all_stress += sheet["stress"][:upto]
            all_end_idx += sheet["phrase_end"][:upto]

        """
        for sheet in sheets:
            #if len(list(set([len(sheet["durations"]), len(sheet["syllables"]), len(sheet["pitches"]), len(sheet["stress"]), len(sheet["phrase_end"])])))>1:
            #    print sheet
            #    raise NotImplementedError
            upto = len(sheet["durations"]) - 1
            all_durations += sheet["durations"][1:upto+1] + [0]
            all_durations_prev += sheet["durations"][:upto] + [0]
            all_syllables += sheet["syllables"][1:upto+1] + ["<end_file>"]
            all_pitches += sheet["pitches"][1:upto+1] + ["R"]
            all_pitches_prev += sheet["pitches"][:upto] + ["R"]
            all_stress += sheet["stress"][1:upto+1] + ["_"]
            all_end_idx += sheet["phrase_end"][1:upto+1] + [1]

        print durations_set
        print len(durations_set)

        print syllables_set
        print len(syllables_set)

        print pitches_set
        print len(pitches_set)

        (all_durations_indices, durations_vocab) = pd.factorize(all_durations_prev + all_durations)
        durations_prev_indices = all_durations_indices[:len(all_durations_indices)/2]
        durations_indices = all_durations_indices[len(all_durations_indices)/2:]
        (syllables_indices, syllables_vocab) = pd.factorize(all_syllables)
        (all_pitches_indices, pitches_vocab) = pd.factorize(all_pitches_prev + all_pitches)
        pitches_prev_indices = all_pitches_indices[:len(all_pitches_indices) / 2]
        pitches_indices = all_pitches_indices[len(all_pitches_indices) / 2:]
        (stress_indices, stress_vocab) = pd.factorize(all_stress)

        instances_num = (len(durations_indices)-1) // sequence_length

        train_data_durations = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_durations_prev = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_syllables = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_pitches = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_pitches_prev = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_stress = np.zeros((instances_num, sequence_length), dtype=np.uint16)
        train_data_phrase_end = np.zeros((instances_num, sequence_length), dtype=np.uint16)

        print len(durations_indices), len(syllables_indices), len(pitches_indices), len(stress_indices), len(all_end_idx)

        for j in range(instances_num):
            for i in range(sequence_length):
                train_data_durations[j][i] = durations_indices[i + j * sequence_length]
                train_data_durations_prev[j][i] = durations_prev_indices[i + j * sequence_length]
                train_data_syllables[j][i] = syllables_indices[i + j * sequence_length]
                train_data_pitches[j][i] = pitches_indices[i + j * sequence_length]
                train_data_pitches_prev[j][i] = pitches_prev_indices[i + j * sequence_length]
                train_data_stress[j][i] = stress_indices[i + j * sequence_length]
                train_data_phrase_end[j][i] = all_end_idx[i + j * sequence_length]

        fout = h5py.File(hdf5_out, mode='w')

        note_durations = fout.create_dataset('durations', train_data_durations.shape, dtype='uint16')
        note_durations_prev = fout.create_dataset('durations_prev', train_data_durations.shape, dtype='uint16')
        note_syllables = fout.create_dataset('syllables', train_data_syllables.shape, dtype='uint16')
        note_pitches = fout.create_dataset('pitches', train_data_syllables.shape, dtype='uint16')
        note_pitches_prev = fout.create_dataset('pitches_prev', train_data_syllables.shape, dtype='uint16')
        note_stress = fout.create_dataset('stress', train_data_stress.shape, dtype='uint16')
        note_end = fout.create_dataset('phrase_end', train_data_phrase_end.shape, dtype='uint16')

        note_durations[...] = train_data_durations
        note_durations_prev[...] = train_data_durations_prev
        note_syllables[...] = train_data_syllables
        note_pitches[...] = train_data_pitches
        note_pitches_prev[...] = train_data_pitches_prev
        note_stress[...] = train_data_stress
        note_end[...] = train_data_phrase_end

        split_num = int(instances_num * 0.9)

        split_dict = {
            'train': {
                'durations': (0, split_num),
                'durations_prev': (0, split_num),
                'syllables': (0, split_num),
                'pitches': (0, split_num),
                'pitches_prev': (0, split_num),
                'stress': (0, split_num),
                'phrase_end': (0, split_num)},
            'test': {
                'durations': (split_num+1, instances_num),
                'durations_prev': (split_num+1, instances_num),
                'syllables': (split_num+1, instances_num),
                'pitches': (split_num+1, instances_num),
                'pitches_prev': (split_num+1, instances_num),
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
    ds = T_H5PYDataset('dataset/normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', which_sets=('train',))
    print ds.syllables_vocab()

#train_data = H5PYDataset('dataset.hdf5', which_sets=('train',), load_in_memory=True)
#test_data  = H5PYDataset('dataset.hdf5', which_sets=('test',), load_in_memory=True)
