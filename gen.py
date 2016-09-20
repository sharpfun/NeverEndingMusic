import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

__author__ = 'Steffen'

from model import MusicNetwork
import music_prepare.dataset
import music_prepare.predict_pitches
import music_prepare.mxml_utils
from datetime import datetime


def sample_musicxml(input_string):
    m = MusicNetwork('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')
    m.load()

    durations, pitches, syllables = m.sample(input_string)

    print syllables

    return music_prepare.mxml_utils.create_mxml(syllables, [''] * len(durations),
                                                durations, pitches)


if __name__ == '__main__':
    mxml = sample_musicxml("""We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us, if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely, but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this in hunger for bread, not in thirst for revenge.""")
    with open('generated/out ' + str(datetime.utcnow()) + '.xml', 'w') as f:
        f.write(mxml)