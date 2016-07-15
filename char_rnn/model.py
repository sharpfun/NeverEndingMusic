__author__ = 'Steffen'

from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks import Tanh, Linear, NDimensionalSoftmax
from blocks import initialization
from blocks.initialization import Constant
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Merge
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.algorithms import GradientDescent, StepClipping, CompositeRule, Adam
from blocks.main_loop import MainLoop
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.saveload import Checkpoint
from blocks.bricks.cost import CategoricalCrossEntropy
import numpy

from theano import tensor

class ParallelCharModel:

    def __init__(self, input1_size, input2_size, lookup1_dim=200, lookup2_dim=200, hidden_size=512):
        self.hidden_size = hidden_size
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.lookup1_dim = lookup1_dim
        self.lookup2_dim = lookup2_dim

        x1 = tensor.lmatrix('durations')
        x2 = tensor.lmatrix('syllables')
        y = tensor.lmatrix('pitches')

        lookup1 = LookupTable(dim=self.lookup1_dim, length=self.input1_size, name='lookup1',
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        lookup1.initialize()
        lookup2 = LookupTable(dim=self.lookup2_dim, length=self.input2_size, name='lookup2',
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        lookup2.initialize()
        merge = Merge(['lookup1', 'lookup2'], [self.lookup1_dim, self.lookup2_dim], self.hidden_size,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        merge.initialize()
        recurrent_block = LSTM(dim=self.hidden_size, activation=Tanh(),
                              weights_init=initialization.Uniform(width=0.01)) #RecurrentStack([LSTM(dim=self.hidden_size, activation=Tanh())] * 3)
        recurrent_block.initialize()
        linear = Linear(input_dim=self.hidden_size, output_dim=self.input1_size,
                              weights_init=initialization.Uniform(width=0.01),
                              biases_init=Constant(0))
        linear.initialize()
        softmax = NDimensionalSoftmax()

        l1 = lookup1.apply(x1)
        l2 = lookup2.apply(x2)
        m = merge.apply(l1, l2)
        h = recurrent_block.apply(m)
        a = linear.apply(h)

        y_hat = softmax.apply(a, extra_ndim=1)
        # ValueError: x must be 1-d or 2-d tensor of floats. Got TensorType(float64, 3D)

        self.Cost = softmax.categorical_cross_entropy(y, a, extra_ndim=1).mean()

        self.ComputationGraph = ComputationGraph(self.Cost)

        self.Model = Model(y_hat)

    def train(self, training_data):

        step_rules = [Adam(), StepClipping(1.0)]

        algorithm = GradientDescent(cost=self.Cost,
                                    parameters=self.ComputationGraph.parameters,
                                    step_rule=CompositeRule(step_rules))

        train_stream = DataStream.default_stream(
            training_data, iteration_scheme=SequentialScheme(
                training_data.num_examples, batch_size=20))

        main = MainLoop(
            model=Model(self.Cost),
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=[
                FinishAfter(),
                Printing(),
                Checkpoint('trainingdata.tar', every_n_epochs=10)
            ])

        main.run()

x = ParallelCharModel(100, 100)
x.train('normalized_syllables_rhythm_notes.json-seqlen-100.hdf5')