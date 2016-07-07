import os
os.environ["THEANO_FLAGS"] = "nvcc.flags=-D_FORCE_INLINES"

from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent
from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks import Tanh
from blocks.roles import WEIGHT
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from theano.sandbox.rng_mrg import MRG_RandomStreams
from numpy import random
from blocks.algorithms import Adam
from blocks.main_loop import MainLoop
from ShowOutput import ShowOutput
from RandomSoftmaxEmitter import RandomSoftmaxEmitter
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.algorithms import GradientDescent, Scale, RMSProp, StepClipping, CompositeRule
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter

from blocks.bricks.cost import BinaryCrossEntropy, MisclassificationRate
import os.path
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.serialization import load

import theano.tensor

import dataset


def train():

    if os.path.isfile('trainingdata.tar'):
        with open('trainingdata.tar', 'rb') as f:
            main = load(f)
    else:
        hidden_size = 512

        train_dataset = dataset.T_H5PYDataset('dataset/wikifonia-seqlen-100.txt.hdf5', which_sets=('train',))

        alphabet_len = train_dataset.vocab_size()

        x = theano.tensor.lmatrix('inchar')

        recurrent_block = LSTM(dim=hidden_size, activation=Tanh())
        recurrent_block2 = LSTM(dim=hidden_size, activation=Tanh())
        recurrent_block3 = LSTM(dim=hidden_size, activation=Tanh())

        transition = RecurrentStack([recurrent_block, recurrent_block2, recurrent_block3])

        readout = Readout(
            readout_dim=alphabet_len,
            feedback_brick=LookupFeedback(alphabet_len, hidden_size, name='feedback'),
            source_names=[thing for thing in transition.apply.states if "states" in thing],
            emitter=RandomSoftmaxEmitter(),
            name='readout'
        )

        gen = SequenceGenerator(readout=readout,
                                transition=transition,
                                weights_init=Uniform(width=0.02),
                                biases_init=Uniform(width=0.0001),
                                name='sequencegenerator')

        gen.push_initialization_config()
        gen.initialize()

        cost = gen.cost(outputs=x)
        cost.name = 'cost'

        cg = ComputationGraph(cost)

        step_rules = [Adam(), StepClipping(1.0)]

        algorithm = GradientDescent(cost=cost,
                                    parameters=cg.parameters,
                                    step_rule=CompositeRule(step_rules),
                                    on_unused_sources='ignore')

        train_stream = DataStream.default_stream(
            train_dataset, iteration_scheme=SequentialScheme(
                train_dataset.num_examples, batch_size=20))

        main = MainLoop(
            model=Model(cost),
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=[
                FinishAfter(),
                Printing(),
                Checkpoint('trainingdata.tar', every_n_epochs=10),
                ShowOutput(every_n_epochs=10)
            ])

    main.run()

if __name__ == "__main__":
    train()
