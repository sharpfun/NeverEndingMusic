__author__ = 'Steffen'

from model import MusicNetwork
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils

m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.load()
dataset = music_prepare.dataset.T_H5PYDataset('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5', which_sets=('train',))

input_str_split =  ("we|are|ac-|count-|ed|poor|cit-|i-|zens|the|pa-|tri-|cians|good|"+\
       "what|au-|thor-|i-|ty|sur-|feits|on|would|re-|lieve|us|if|they|"+\
       "would|yield|us|but|the|su-|per-|flu-|i-|ty|while|it|were|"+\
       "whole-|some|we|might|guess|they|re-|lieved|us|hu-|mane-|ly|"+\
       "but|they|think|we|are|too|dear|the|lean-|ness|that|"+\
       "af-|flicts|us|the|ob-|ject|of|our|mis-|er-|y|is|as|an|"+\
       "in-|ven-|to-|ry|to|par-|tic-|u-|lar-|ise|their|a-|bun-|dance|our|"+\
       "suf-|fer-|ance|is|a|gain|to|them|Let|us|re-|venge|this|with|"+\
       "our|pikes|ere|we|be-|come|rakes|for|the|gods|know|i|"+\
       "speak|this|in|hunger|for|bread|not|in|thirst|for|re-|venge|").split('|')
input_syllables = dataset.syllables_encode(input_str_split)
input_syllables_decoded = dataset.syllables_decode(input_syllables)

durations = [6720, 26880, 13440, 13440, 13440, 13440, 13440, 6720, 13440, 6720, 13440, 20160, 80640, 13440, 20160, 26880, 13440, 26880, 13440, 26880, 53760, 26880, 13440, 13440, 13440, 13440, 26880, 26880, 26880, 13440, 26880, 13440, 13440, 13440, 26880, 26880, 26880, 26880, 26880, 26880, 26880, 53760, 53760, 26880, 26880, 26880, 13440, 13440, 26880, 40320, 13440, 26880, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 26880, 13440, 13440, 26880, 13440, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 26880, 13440, 13440, 40320, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 26880, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 13440, 6720, 13440, 6720, 6720, 20160, 6720, 6720, 13440, 13440]
input_durations = dataset.durations_encode(durations)

sampled = m.sample_pitch(input_syllables, input_durations)

data = dataset.pitches_decode(sampled)

mxml = music_prepare.mxml_utils.create_mxml(input_str_split, input_syllables_decoded,
                                            durations,
                                            dataset.pitches_decode(sampled))

with open('out.xml', 'w') as f:
    f.write(mxml)