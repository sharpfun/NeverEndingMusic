__author__ = 'Steffen'

from blocks.serialization import load
from blocks.model import ComputationGraph
from CharCorpusDataset import HDF5CharEncoder
from RandomSoftmaxEmitter import RandomSoftmaxEmitter

import dataset
from dataset import Corpus

def gen(num_sample=1000,num_lines=20):
    c = Corpus(open('lyrics_out.txt').read())

    with open('trainingdata.tar', 'rb') as f:
        model = load(f).model

    generator = model.top_bricks[0]

    sample = ComputationGraph(generator.generate(
        n_steps=num_sample,
        batch_size=1,
        iterate=True
    )).get_theano_function()

    output_char_int = sample()[6][:,0]

    output = "\n".join(("".join(c.decode(output_char_int))).splitlines()[0:num_lines])
    return output

if __name__ == "__main__":
    print gen()

