__author__ = 'Steffen'

from model import MusicNetwork
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils

m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
m.load()

durations, pitches, syllables = m.sample("Happy birthday to you, happy birthday to you, happy birthday dear [Name Here], happy birthday to you.")

mxml = music_prepare.mxml_utils.create_mxml(syllables, [''] * len(durations),
                                            durations, pitches)

with open('out.xml', 'w') as f:
    f.write(mxml)