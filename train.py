__author__ = 'Steffen'

from model import MusicNetwork

m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.RhythmModel.train('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', 'trainingdata_rhythm.tar', n_epochs=200)
m.PitchModel.train('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', 'trainingdata_pitches.tar', n_epochs=200)