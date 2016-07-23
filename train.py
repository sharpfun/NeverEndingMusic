__author__ = 'Steffen'

from model import MusicNetwork

m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.RhythmModel.train('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', 'trainingdata_rhythm.tar')
m.PitchModel.train('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', 'trainingdata_pitches.tar')