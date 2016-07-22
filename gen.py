__author__ = 'Steffen'

from model import MusicRNNModel
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils

m = MusicRNNModel('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.load()
dataset = music_prepare.dataset.T_H5PYDataset('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', which_sets=('train',))

input_str_split = ["leave", "me", "out", "with", "the", "waste", "this", "is", "not", "what", "i", "do", "its", "the", "wrong", "kind", "of", "place", "to", "be", "think-", "king", "of", "you", "its", "the", "wrong", "time", "for", "some-", "bo-", "dy", "new", "its", "a", "small", "crime", "and", "ive", "got", "no", "ex-", "cuse", "is", "that", "al-", "right", "yeah", "give", "my", "gun", "a-", "way", "when", "its", "loa-", "ded", "is", "that", "all-", "right", "yeah", "if", "you", "dont", "shoot", "how", "am", "i", "sup-", "posed", "to", "hold", "it", "is", "that", "all", "right", "yeah", "give", "my", "gun", "a-", "way", "when", "its", "loa-", "ded", "is", "that", "all-", "right", "yeah", "with", "you", "leave", "me", "out"]
input_syllables = dataset.syllables_encode(input_str_split)
input_syllables_decoded = dataset.syllables_decode(input_syllables)

durations = [13440, 8960, 8960, 53760, 13440, 8960, 8960, 53760, 13440, 8960, 8960, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 40320, 13440, 8960, 8960, 53760, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 26880, 26880, 26880, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 13440, 8960, 8960, 13440, 40320, 13440, 13440, 13440, 13440, 13440, 8960, 8960, 53760, 13440, 8960, 8960, 53760, 13440, 8960, 8960, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 40320, 13440, 8960, 8960, 53760, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 26880, 26880, 26880, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 13440, 8960, 8960, 13440, 40320, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 13440, 40320, 13440, 13440, 13440, 13440, 13440, 26880, 26880, 13440, 40320, 26880, 13440, 40320, 13440, 8960, 8960, 8960, 13440, 8960, 8960, 8960, 13440, 8960, 8960, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 40320, 13440, 8960, 8960, 8960, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 26880, 26880, 26880, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 26880, 13440, 13440, 26880, 13440, 8960, 8960, 13440, 40320, 13440, 13440, 13440, 13440]
input_durations = dataset.durations_encode(durations)

print len(input_syllables)
print len(input_durations)

sampled = m.sample_pitch(input_syllables, input_durations)

data = dataset.pitches_decode(sampled)

mxml = music_prepare.mxml_utils.create_mxml([""] * len(durations), [""] * len(durations),
                                            durations,
                                            dataset.pitches_decode(sampled))

with open('out.xml', 'w') as f:
    f.write(mxml)