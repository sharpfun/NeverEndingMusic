__author__ = 'Steffen'

from model import MusicRNNModel

m = MusicRNNModel('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.initialize_pitch_model()
m.train_pitch()