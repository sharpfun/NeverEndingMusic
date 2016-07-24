__author__ = 'Steffen'

from model import MusicNetwork
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils


def sample_musicxml(input_string):
    m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
    m.load()

    durations, pitches, syllables = m.sample(input_string)

    return music_prepare.mxml_utils.create_mxml(syllables, [''] * len(durations),
                                                durations, pitches)


if __name__ == '__main__':
    mxml = sample_musicxml("Happy birthday to you, happy birthday to you, happy birthday dear [Name Here], happy birthday to you.")
    with open('out.xml', 'w') as f:
        f.write(mxml)