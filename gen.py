__author__ = 'Steffen'

from model import MusicRNNModel
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils

m = MusicRNNModel('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.load()
dataset = music_prepare.dataset.T_H5PYDataset('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', which_sets=('train',))

input_str = "love-looks-not-with-the-eyes-but-with-the-mind-and-there-fore-is-winged-cupid-pain-ted-blind"
input_str_split = input_str.split("-")
input_syllables = dataset.syllables_encode(input_str_split)
input_syllables_decoded = dataset.syllables_decode(input_syllables)

durations = [13440, 26880, 17920, 13440, 26880, 26880, 26880, 13440, 13440, 80640, 13440, 13440, 26880, 53760, 26880, 26880, 13440, 13440, 13440]
input_durations = dataset.durations_encode(durations)


sampled = m.sample_pitch([0] * len(input_durations), input_durations)

data = dataset.pitches_decode(sampled)

mxml = music_prepare.mxml_utils.create_mxml(input_str_split, input_syllables_decoded,
                                            dataset.durations_decode(input_durations),
                                            dataset.pitches_decode(sampled))

with open('out.xml', 'w') as f:
    f.write(mxml)